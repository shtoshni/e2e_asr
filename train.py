from __future__ import absolute_import
from __future__ import division

import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import random
import sys
import time

import numpy as np
from six.moves import xrange
import cPickle as pickle
import argparse
import operator
from bunch import bunchify
import editdistance as ed
import glob

import data_utils
import seq2seq_model
import subprocess

import tensorflow as tf
from tensorflow.python.ops import variable_scope

from seq2seq_model import Seq2SeqModel
from encoder import Encoder
from attn_decoder import AttnDecoder

#import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()


_buckets = [(210, (60, 50)), (346, (120, 110)), (548, (180, 140)), (850, (200, 150)), (1500, (380, 250))]
bucket_size = [input_size for input_size, _ in _buckets]
task_to_freq = {'phone':1.0, 'char':1.0}
eval_task_to_id = {'char':0}

NUM_THREADS = 1

FLAGS = object()

def parse_tasks(task_string):
    tasks = ["char"]
    if "p" in tasks:
        tasks.append("phone")
    return tasks


def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("-bsize", "--batch_size", default=64, type=int, help="Mini-batch Size")
    parser.add_argument("-hsize", "--hidden_size", default=256, type=int, help="Hidden layer size")
    parser.add_argument("-hsize_decoder", "--hidden_size_decoder", default=256, type=int, help="Hidden layer size")
    parser.add_argument("-nlp", "--num_layers_phone", default=1, type=int, help="Number of layers to decode side")
    parser.add_argument("-nlc", "--num_layers_char", default=2, type=int, help="Number of layers to decode side")

    parser.add_argument("-tvp", "--target_vocab_file_phone", default="phone.txt", type=str, help="Vocab file for phone target")
    parser.add_argument("-tvc", "--target_vocab_file_char", default="char.txt", type=str, help="Vocab file for character target")

    parser.add_argument("-vocab_dir", "--vocab_dir", default="/share/data/speech/shtoshni/research/asr_multi/data/vocab", type=str, help="Vocab directory")
    #parser.add_argument("-data_dir", "--data_dir", default="/share/data/speech/shtoshni/research/asr_multi/data/ctc_data", type=str, help="Data directory")
    #parser.add_argument("-tb_dir", "--train_base_dir", default="/share/data/speech/shtoshni/research/asr_multi/models", type=str, help="Training directory")
    #parser.add_argument("-bm_dir", "--best_model_dir", default="/share/data/speech/shtoshni/research/asr_multi/models/best_models", type=str, help="Training directory")
    parser.add_argument("-data_dir", "--data_dir", default="/scratch/asr_multi/data/ctc_data", type=str, help="Data directory")
    parser.add_argument("-tb_dir", "--train_base_dir", default="/scratch/asr_multi/models", type=str, help="Training directory")
    parser.add_argument("-bm_dir", "--best_model_dir", default="/scratch/asr_multi/models/best_models", type=str, help="Training directory")
    parser.add_argument("-tasks", "--tasks", default="", type=str, help="Auxiliary task choices")

    parser.add_argument("-skip_step", "--skip_step", default=1, type=int, help="Frame skipping factor as we go up the stacked layers")

    parser.add_argument("-out_prob", "--output_keep_prob", default=0.9, type=float, help="Output keep probability for dropout")
    parser.add_argument("-feat_len", "--feat_length", default=80, type=int, help="Number of features per frame")
    parser.add_argument("-base_pyramid", "--base_pyramid", default=False, action="store_true", help="Do pyramid at feature level as well?")
    parser.add_argument("-sch_samp", "--sch_samp", default=True, action="store_true", help="Do pyramid at feature level as well?")

    parser.add_argument("-avg", "--avg", default=False, action="store_true", help="Average the loss")
    parser.add_argument("-num_files", "--num_files", default=0, type=int, help="Num files")
    parser.add_argument("-max_epochs", "--max_epochs", default=500, type=int, help="Max epochs")
    parser.add_argument("-eval", "--eval_dev", default=False, action="store_true", help="Get dev set results using the last saved model")
    parser.add_argument("-test", "--test", default=False, action="store_true", help="Get test results using the last saved model")
    parser.add_argument("-run_id", "--run_id", default=0, type=int, help="Run ID")

    args = parser.parse_args()
    arg_dict = vars(args)

    arg_dict['tasks'] = parse_tasks(arg_dict['tasks'])
    arg_dict['steps_per_checkpoint'] = 100

    skip_string = ""
    if arg_dict['skip_step'] != 1:
        skip_string = "skip_" + str(arg_dict['skip_step']) + "_"

    samp_string = ""
    if arg_dict['sch_samp'] != False:
        samp_string = "samp_"

    num_layer_string = ""
    for task in arg_dict['tasks']:
        num_layer_string += 'nl' + task + '_' + str(arg_dict['num_layers_' + task]) + '_'


    train_dir = (skip_string +
                samp_string +
                num_layer_string +
                'out_prob_' + str(arg_dict['output_keep_prob']) + '_' +
                'run_id_' + str(arg_dict['run_id']) +
                ('_avg_' if arg_dict['avg'] else '')
    )

    arg_dict['train_dir'] = os.path.join(arg_dict['train_base_dir'], train_dir)
    arg_dict['best_model_dir'] = os.path.join(arg_dict['best_model_dir'], train_dir)

    arg_dict['num_layers'] = {}
    for task in arg_dict['tasks']:
        arg_dict['num_layers'][task] = arg_dict['num_layers_' + task]

    arg_dict['target_vocab_file'] = {}
    for task in arg_dict['tasks']:
        arg_dict['target_vocab_file'][task] = arg_dict['target_vocab_file_' + task]

    arg_dict['output_vocab_size'] = {}
    for task in arg_dict['tasks']:
        target_vocab, _ = data_utils.initialize_vocabulary(os.path.join(arg_dict['vocab_dir'], \
                arg_dict['target_vocab_file'][task]))
        arg_dict['output_vocab_size'][task] = len(target_vocab)

    if arg_dict['test'] and arg_dict['eval_dev']:
        arg_dict['apply_dropout'] = False

    if not arg_dict['test'] and not arg_dict['eval_dev']:
        if not os.path.exists(arg_dict['train_dir']):
            os.makedirs(arg_dict['train_dir'])
        if not os.path.exists(arg_dict['best_model_dir']):
            os.makedirs(arg_dict['best_model_dir'])

        ## Sort the arg_dict to create a parameter file
        parameter_file = 'parameters.txt'
        sorted_args = sorted(arg_dict.items(), key=operator.itemgetter(0))

        with open(os.path.join(arg_dict['train_dir'], parameter_file), 'w') as g:
            for arg, arg_val in sorted_args:
                sys.stdout.write(arg + "\t" + str(arg_val) + "\n")
                sys.stdout.flush()
                g.write(arg + "\t" + str(arg_val) + "\n")

    global FLAGS
    FLAGS = bunchify(arg_dict)


def load_dev_data(test=False):
    if not test:
        dev_data_path = os.path.join(FLAGS.data_dir, 'dev_short.pickle')
    else:
        dev_data_path = os.path.join(FLAGS.data_dir, 'eval.pickle')
    dev_set = pickle.load(open(dev_data_path))
    num_dev_samples = sum([len(bucket_instances) for bucket_instances in dev_set])
    print ("Number of dev samples: %d" %num_dev_samples)
    return dev_set


def get_train_data():
    all_files = glob.glob(FLAGS.data_dir + "/train*")
    train_files = []
    for file_name in all_files:
        #batch_idx = int(os.path.basename(file_name).split(".")[1])
        train_files.append(file_name)
    train_files.sort()
    if FLAGS.num_files > 0:
        train_files = train_files[:FLAGS.num_files]

    #print (train_files)
    print (len(train_files))
    number_of_instances = 0
    for train_file in train_files:
        print(train_file)
        sys.stdout.flush()
        number_of_instances += sum([1 for _ in tf.python_io.tf_record_iterator(train_file)])
        ## Using ceil below since we allow for smaller final batch

    number_of_batches = int(np.ceil(number_of_instances/float(FLAGS.batch_size)))
    return train_files, number_of_batches


def create_seq2seq_model(isTraining, num_layers, queue=None):
    seq2seq_params = Seq2SeqModel.class_params()
    seq2seq_params.isTraining = isTraining
    seq2seq_params.num_layers = num_layers
    seq2seq_params.tasks = list(num_layers.keys())

    encoder = create_encoder(isTraining)
    decoder = {}
    for task in num_layers:
        # Create separate decoders for separate tasks
        with variable_scope.variable_scope(task):
            decoder[task] = create_decoder(isTraining)

    seq2seq_model = Seq2SeqModel(encoder, decoder, params=seq2seq_params, queue=queue)
    return seq2seq_model


def create_encoder(isTraining):
    if isTraining:
        encoder = Encoder()
    else:
        encoder_params = Encoder.class_params()
        encoder_params.isTraining = False
        encoder = Encoder(encoder_params)
    return encoder


def create_decoder(isTraining):
    if isTraining:
        decoder = AttnDecoder()
    else:
        decoder_params = AttnDecoder.class_params()
        decoder_params.isTraining = False
        decoder = AttnDecoder(decoder_params)
    return decoder


def create_model(session, isTraining, num_layers, model_path=None, queue=None, actual_eval=False):
    """Create model and initialize or load parameters in session."""
    model = create_seq2seq_model(isTraining, num_layers, queue=queue)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    ckpt_best = tf.train.get_checkpoint_state(FLAGS.best_model_dir)
    if ckpt:
        steps_done = int(ckpt.model_checkpoint_path.split('-')[-1])
        if ckpt_best:
            steps_done_best = int(ckpt_best.model_checkpoint_path.split('-')[-1])
            if (steps_done_best > steps_done) or actual_eval:
                ckpt = ckpt_best
                steps_done = steps_done_best
        print("loaded from %d done steps" %(steps_done) )
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        steps_done = int(ckpt.model_checkpoint_path.split('-')[-1])
        print("loaded from %d done steps" %(steps_done) )
        sys.stdout.flush()
    elif ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path) and model_path is not None:
        model.saver.restore(session, model_path)
        steps_done = int(model_path.split('-')[-1])
        print("Reading model parameters from %s" % model_path)
        print("loaded from %d done steps" %(steps_done) )
        sys.stdout.flush()
    else:
        print("Created model with fresh parameters.")
        sys.stdout.flush()
        session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        steps_done = 0
    return model, steps_done


def train():
    """Train a sequence to sequence parser."""
    char_vocab_path = os.path.join(FLAGS.vocab_dir, FLAGS.target_vocab_file['char'])
    char_vocab, rev_char_vocab = data_utils.initialize_vocabulary(char_vocab_path)
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
        print("Loading train data from %s" % FLAGS.data_dir)
        train_files, num_batch = get_train_data()
        print ("Number of minibatches: %d" %(num_batch))
        bucket_queue = tf.train.string_input_producer(train_files, shuffle=True)

        # Create model.
        print("Creating %d layers of %d units." % (max(FLAGS.num_layers.values()), FLAGS.hidden_size))
        sys.stdout.flush()
        with tf.variable_scope("model", reuse=None):
            model, steps_done = create_model(sess, True, num_layers=FLAGS.num_layers, queue=bucket_queue)
        with tf.variable_scope("model", reuse=True):
            print ("Creating dev model")
            model_dev  = create_seq2seq_model(isTraining=False, num_layers={'char': FLAGS.num_layers['char']})


        # Prepare training data
        epoch = model.epoch.eval()
        epochs_left = FLAGS.max_epochs - epoch

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Test dev results
        dev_set = load_dev_data()
        #asr_err_best = 1
        asr_err_best = asr_decode(model_dev, sess, dev_set)
        if steps_done > 0:
            ## Some training has been done
            score_file = os.path.join(FLAGS.train_dir, "best.txt")
            ## Check existence of such a file
            if os.path.isfile(score_file):
                try:
                    asr_err_best = float(open(score_file).readline().strip("\n"))
                except ValueError:
                    asr_err_best = 1

        print ("Best ASR error rate - %f" %asr_err_best)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        total_minibatches = num_batch
        train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/train', tf.get_default_graph())

        while epoch <= FLAGS.max_epochs:
            try:
                while not coord.should_stop():
                    print("Epochs done: %d" %epoch)
                    sys.stdout.flush()

                    for i in xrange(total_minibatches):
                        task = "char"
                        start_time = time.time()
                        output_feed = [model.updates,  # Update Op that does SGD.
                                       model.losses]  # Loss for this batch.

                        current_step += 1
                        if (current_step % FLAGS.steps_per_checkpoint) == 0:
                            output_feed.append(model.merged)
                            _, step_loss, train_summary = sess.run(output_feed)
                            train_writer.add_summary(train_summary, current_step)
                        else:
                            _, step_loss = sess.run(output_feed)
                        step_loss = step_loss[task]

                        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                        loss += step_loss / FLAGS.steps_per_checkpoint

                        if current_step % FLAGS.steps_per_checkpoint == 0:
                            # Print statistics for the previous epoch.
                            perplexity = math.exp(loss) if loss < 300 else float('inf')

                            print ("global step %d learning rate %.4f step-time %.2f perplexity "
                                   "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                             step_time, perplexity))

                            asr_err_cur = asr_decode(model_dev, sess, dev_set)
                            print ("ASR error: %.4f" %(asr_err_cur))

                            err_summary = get_summary(asr_err_cur, "ASR Error")
                            train_writer.add_summary(err_summary, current_step)

                            if (len(previous_losses) > 0 and loss > previous_losses[-1]):
                                if model.learning_rate.eval() > 1e-4:
                                    sess.run(model.learning_rate_decay_op)
                                    print ("Learning rate decreased !!")
                            previous_losses.append(loss)

                            ## Early stopping - ONLY UPDATING MODEL IF BETTER PERFORMANCE ON DEV
                            if asr_err_best > asr_err_cur:
                                asr_err_best = asr_err_cur
                                # Save model
                                print("Best ASR Error rate: %.4f" % asr_err_best)
                                print("Saving the best model !!")

                                # Save the best score
                                f = open(os.path.join(FLAGS.train_dir, "best.txt"), "w")
                                f.write(str(asr_err_best))
                                f.close()

                                ## Save the model in best model directory
                                checkpoint_path = os.path.join(FLAGS.best_model_dir, "asr.ckpt")
                                model.best_saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)
                                ## Also save the model for plotting
                                checkpoint_path = os.path.join(FLAGS.train_dir, "asr.ckpt")
                                model.saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)
                            else:
                                ## Save the model in regular directory
                                print("Saving for the sake of record - huh")
                                checkpoint_path = os.path.join(FLAGS.train_dir, "asr.ckpt")
                                model.saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)


                            print ("\n")
                            sys.stdout.flush()
                            step_time, loss = 0.0, 0.0

                    # Update epoch counter
                    sess.run(model.epoch_incr)
                    epoch += 1

            except Exception as e:
                coord.request_stop(e)
                coord.join(threads)
                break

        coord.request_stop()
        coord.join(threads)


def get_summary(value, tag):
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])


def get_dev_cross_entropy(sess, model_dev, dev_set):
    total_chars = 0.0
    log_loss = 0
    clsfcn_loss = 0 ## classification loss
    for bucket_id in xrange(len(dev_set)):
        bucket_size = len(dev_set[bucket_id])
        offsets = np.arange(0, bucket_size, FLAGS.batch_size)
        for batch_offset in offsets:
            all_examples = dev_set[bucket_id][batch_offset:batch_offset+FLAGS.batch_size]
            encoder_inputs, decoder_inputs,  seq_len, seq_len_target = \
                    model_dev.get_batch({bucket_id: all_examples}, bucket_id, task='char', do_eval=False)


            frac_errs, normalized_loss, minibatch_chars = model_dev.step(
                sess, encoder_inputs, seq_len, decoder_inputs, seq_len_target)

            total_chars += minibatch_chars
            clsfcn_loss += frac_errs * minibatch_chars
            log_loss += normalized_loss * minibatch_chars

    avg_log_loss = log_loss/total_chars
    avg_clsfcn_loss = clsfcn_loss/total_chars

    return avg_log_loss, avg_clsfcn_loss


def asr_decode(model_dev, sess, dev_set):
    # Load vocabularies.
    char_vocab_path = os.path.join(FLAGS.vocab_dir, FLAGS.target_vocab_file['char'])
    char_vocab, rev_char_vocab = data_utils.initialize_vocabulary(char_vocab_path)

    gold_asr_file = os.path.join(FLAGS.train_dir, 'gold_asr.txt')
    decoded_asr_file = os.path.join(FLAGS.train_dir, 'decoded_asr.txt')
    raw_asr_file = os.path.join(FLAGS.train_dir, 'raw_asr.txt')

    fout_gold = open(gold_asr_file, 'w')
    fout_raw_asr = open(raw_asr_file, 'w')
    fout_asr = open(decoded_asr_file, 'w')

    num_dev_sents = 0
    total_errors = 0
    total_words = 0
    ## Set numpy printing threshold
    np.set_printoptions(threshold=np.inf)
    batch_size=FLAGS.batch_size
    for bucket_id in xrange(len(dev_set)):
        bucket_size = len(dev_set[bucket_id])
        offsets = np.arange(0, bucket_size, batch_size)
        for batch_offset in offsets:
            all_examples = dev_set[bucket_id][batch_offset:batch_offset+batch_size]

            model_dev.params.batch_size = len(all_examples)
            log_mels = [x[0] for x in all_examples]
            gold_ids = [x[1] for x in all_examples]
            sent_id_vals = [x[2] for x in all_examples]
            dec_ids = [[]] * len(gold_ids)

            encoder_inputs, decoder_inputs,  seq_len, seq_len_target = \
                model_dev.get_batch({bucket_id: zip(log_mels, gold_ids)}, bucket_id, task='char', do_eval=True)


            _, output_logits = model_dev.step(sess, encoder_inputs, seq_len,
                                              decoder_inputs, seq_len_target)

            outputs = np.argmax(output_logits, axis=1)
            outputs = np.reshape(outputs, (max(seq_len_target['char']), model_dev.params.batch_size)) ##T*B

            to_decode = np.array(outputs).T ## T * B and the transpose makes it B*T

            num_dev_sents += to_decode.shape[0]
            for sent_id in xrange(to_decode.shape[0]):
                asr_out = list(to_decode[sent_id, :])
                if data_utils.EOS_ID in asr_out:
                    asr_out = asr_out[:asr_out.index(data_utils.EOS_ID)]

                decoded_asr = ''
                for output in asr_out:
                    decoded_asr += tf.compat.as_str(rev_char_vocab[output])

                gold_asr = ''.join([tf.compat.as_str(rev_char_vocab[output]) for output in gold_ids[sent_id]])
                raw_asr_words, decoded_words = data_utils.get_relevant_words(decoded_asr)
                _, gold_words = data_utils.get_relevant_words(gold_asr)

                total_errors += ed.eval(gold_words, decoded_words)
                total_words += len(gold_words)

                fout_gold.write('{}\n'.format(' '.join(gold_words)))
                fout_raw_asr.write(sent_id_vals[sent_id] + "\t" + '{}\n'.format(' '.join(raw_asr_words)))
                fout_asr.write(sent_id_vals[sent_id] + "\t" + '{}\n'.format(' '.join(decoded_words)))

    # Write to file
    fout_gold.close()
    fout_raw_asr.close()
    fout_asr.close()
    try:
        score = float(total_errors)/float(total_words)
    except ZeroDivisionError:
        score = 0.0

    return score


def decode(test=False):
    """ Decode file sentence-by-sentence  """
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
        # Create model and load parameters.
        with tf.variable_scope("model"):
            model_dev, steps_done = create_model(sess, forward_only=True, task_to_id=eval_task_to_id, actual_eval=True)

        print ("Epochs done: %d" %model_dev.epoch.eval())
        dev_set = load_dev_data(test=test)
        print ("Dev set loaded !!")

        start_time = time.time()
        asr_decode(model_dev, sess, dev_set)
        time_elapsed = time.time() - start_time
        print("Decoding all dev time: ", time_elapsed)


if __name__ == "__main__":
    parse_options()
    if FLAGS.eval_dev:
        decode()
    elif FLAGS.test:
        decode(test=True)
    else:
        train()

