# coding: utf-8

from __future__ import absolute_import
from __future__ import division

import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import copy
import random
import sys
import time

import argparse
import operator
import glob
import re
from datetime import timedelta

import numpy as np
from bunch import Bunch, bunchify
import editdistance as ed
import tensorflow as tf

import data_utils
from attn_decoder import AttnDecoder
from encoder import Encoder
from lm_encoder import LM
from lm_model import LMModel
from seq2seq_model import Seq2SeqModel
from speech_dataset import SpeechDataset
from lm_dataset import LMDataset


NUM_THREADS = 1
FLAGS = object()

def parse_tasks(task_string):
    print (task_string)
    tasks = ["char"]
    if "p" in task_string:
        tasks.append("phone")
    return tasks


def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("-hsize", "--hidden_size", default=256, type=int, help="Hidden layer size")
    parser.add_argument("-hsize_decoder", "--hidden_size_decoder", default=256, type=int, help="Hidden layer size")
    parser.add_argument("-nlp", "--num_layers_phone", default=3, type=int, help="Number of layers to decode side")
    parser.add_argument("-nlc", "--num_layers_char", default=4, type=int, help="Number of layers to decode side")

    parser.add_argument("-tvp", "--target_vocab_file_phone", default="phone.vocab", type=str, help="Vocab file for phone target")
    parser.add_argument("-tvc", "--target_vocab_file_char", default="char.vocab", type=str, help="Vocab file for character target")

    parser.add_argument("-vocab_dir", "--vocab_dir", default="/scratch2/asr_multi/data/lang/vocab", type=str, help="Vocab directory")
    #parser.add_argument("-data_dir", "--data_dir", default="/share/data/speech/shtoshni/research/asr_multi/data/ctc_data", type=str, help="Data directory")
    #parser.add_argument("-tb_dir", "--train_base_dir", default="/share/data/speech/shtoshni/research/asr_multi/models", type=str, help="Training directory")
    #parser.add_argument("-bm_dir", "--best_model_dir", default="/share/data/speech/shtoshni/research/asr_multi/models/best_models", type=str, help="Training directory")
    #parser.add_argument("-data_dir", "--data_dir", default="/scratch/asr_multi/data/ctc_data", type=str, help="Data directory")
    parser.add_argument("-data_dir", "--data_dir", default="/scratch2/asr_multi/data/tfrecords", type=str, help="Data directory")
    parser.add_argument("-lm_data_dir", "--lm_data_dir", default="/scratch2/asr_multi/data/tfrecords/fisher/red_0.7", type=str, help="Data directory")
    parser.add_argument("-tb_dir", "--train_base_dir", default="/scratch2/asr_multi/models", type=str, help="Training directory")
    parser.add_argument("-bm_dir", "--best_model_dir", default="/scratch2/asr_multi/models/best_models", type=str, help="Training directory")
    parser.add_argument("-tasks", "--tasks", default="", type=str, help="Auxiliary task choices")

    parser.add_argument("-skip_step", "--skip_step", default=1, type=int, help="Frame skipping factor as we go up the stacked layers")

    parser.add_argument("-out_prob", "--output_keep_prob", default=0.9, type=float, help="Output keep probability for dropout")
    parser.add_argument("-base_pyramid", "--base_pyramid", default=False, action="store_true", help="Do pyramid at feature level as well?")
    parser.add_argument("-sch_samp", "--sch_samp", default=True, action="store_true", help="Do pyramid at feature level as well?")

    parser.add_argument("-bsize", "--batch_size", default=64, type=int, help="Mini-batch Size")
    parser.add_argument("-feat_len", "--feat_length", default=80, type=int, help="Number of features per frame")
    parser.add_argument("-max_out", "--max_output", default=120, type=int, help="Maximum length of output sequence")

    parser.add_argument("-avg", "--avg", default=False, action="store_true", help="Average the loss")
    parser.add_argument("-steps_per_checkpoint", "--steps_per_checkpoint", default=500,
                        type=int, help="Gradient steps per checkpoint")
    parser.add_argument("-min_epochs", "--min_epochs", default=5, type=int, help="Min epochs BEFORE DECREASING LEARNING RATE")
    parser.add_argument("-max_epochs", "--max_epochs", default=30, type=int, help="Max epochs")
    parser.add_argument("-eval", "--eval_dev", default=False, action="store_true", help="Get dev set results using the last saved model")
    parser.add_argument("-test", "--test", default=False, action="store_true", help="Get test results using the last saved model")
    parser.add_argument("-run_id", "--run_id", default=0, type=int, help="Run ID")

    args = parser.parse_args()
    arg_dict = vars(args)

    arg_dict['tasks'] = parse_tasks(arg_dict['tasks'])

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

        # Sort the arg_dict to create a parameter file
        parameter_file = 'parameters.txt'
        sorted_args = sorted(arg_dict.items(), key=operator.itemgetter(0))

        with open(os.path.join(arg_dict['train_dir'], parameter_file), 'w') as g:
            for arg, arg_val in sorted_args:
                sys.stdout.write(arg + "\t" + str(arg_val) + "\n")
                sys.stdout.flush()
                g.write(arg + "\t" + str(arg_val) + "\n")

    dataset_params = Bunch()
    dataset_params.data_dir = arg_dict['data_dir']
    dataset_params.batch_size = arg_dict['batch_size']
    dataset_params.max_output = arg_dict['max_output']
    dataset_params.feat_length = arg_dict['feat_length']

    global FLAGS
    FLAGS = bunchify(arg_dict)
    FLAGS.dataset_params = dataset_params


def get_split_files(data_dir, split="train"):
    sep = ("" if data_dir[-1] == "/" else "/")
    all_files = glob.glob(data_dir + sep + split + "*")
    #all_files = glob.glob(FLAGS.data_dir + "/train*1.1.*")
    split_files = []
    for file_name in all_files:
        split_files.append(file_name)
    split_files.sort()
    print ("Number of %s files: %d" %(split, len(split_files)))
    return split_files


def create_seq2seq_model(isTraining, num_layers, data_iter):
    seq2seq_params = Seq2SeqModel.class_params()
    seq2seq_params.isTraining = isTraining
    seq2seq_params.num_layers = num_layers
    seq2seq_params.tasks = list(num_layers.keys())

    encoder = create_encoder(isTraining)
    decoder = {}
    for task in num_layers:
        # Create separate decoders for separate tasks
        decoder[task] = create_decoder(isTraining,
                                       vocab_size=FLAGS.output_vocab_size[task],
                                       scope=task)

    seq2seq_model = Seq2SeqModel(encoder, decoder, params=seq2seq_params, data_iter=data_iter)
    return seq2seq_model


def create_encoder(isTraining):
    if isTraining:
        encoder = Encoder()
    else:
        encoder_params = Encoder.class_params()
        encoder_params.isTraining = False
        encoder = Encoder(encoder_params)
    return encoder


def create_decoder(isTraining, vocab_size, scope=None):
    decoder_params = AttnDecoder.class_params()
    decoder_params.isTraining = isTraining
    decoder_params.vocab_size = vocab_size
    decoder = AttnDecoder(decoder_params, scope=scope)
    return decoder


def create_lm_model(vocab_size, encoder_hidden_size, data_iter, scope=None):
    lm_params = LM.class_params()
    lm_params.vocab_size = vocab_size
    lm_params.encoder_hidden_size = 2 * encoder_hidden_size
    lm_encoder = LM(lm_params, scope=scope)

    lm_model = LMModel(lm_encoder, data_iter, LMModel.class_params())

    return lm_model

def create_model(session, isTraining, num_layers, data_iter, model_path=None, actual_eval=False):
    """Create model and initialize or load parameters in session."""
    model = create_seq2seq_model(isTraining, num_layers, data_iter)
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
        steps_done = int(ckpt.model_checkpoint_path.split('-')[-1])
        print("loaded from %d done steps" %(steps_done) )
        sys.stdout.flush()
    else:
        print("Created model with fresh parameters.")
        sys.stdout.flush()
        steps_done = 0
    return model, steps_done


def train():
    """Train a sequence to sequence parser."""
    with tf.Graph().as_default():
        tf.set_random_seed(10)
        with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
            print("Loading train data from %s" % FLAGS.data_dir)
            sys.stdout.flush()

            #train_set = SpeechDataset(FLAGS.dataset_params, "train", isTraining=True)
            #buck_batch_size = [128, 128, 64, 32, 16]
            buck_batch_size = [128, 64, 64, 32, 16]
            buck_train_sets = []
            for batch_id, batch_size in enumerate(buck_batch_size):
                dataset_params = copy.deepcopy(FLAGS.dataset_params)
                dataset_params.batch_size = batch_size
                cur_train_files = get_split_files(FLAGS.data_dir, "train*." + str(batch_id) + ".*.")
                cur_train_set = SpeechDataset(dataset_params, cur_train_files, isTraining=True)
                buck_train_sets.append(cur_train_set)

            iterator = tf.data.Iterator.from_structure(buck_train_sets[0].data_set.output_types,
                                                       buck_train_sets[0].data_set.output_shapes)
            iter_init_list = []
            for train_set in buck_train_sets:
                iter_init_list.append(iterator.make_initializer(train_set.data_set))

            dev_set = SpeechDataset(FLAGS.dataset_params, get_split_files(FLAGS.data_dir, split="dev"),
                                    isTraining=False)

            lm_files = get_split_files(FLAGS.lm_data_dir, split="lm")
            lm_set = LMDataset(lm_files, FLAGS.batch_size)

            with tf.variable_scope("model", reuse=None):
                model, steps_done = create_model(sess, True, FLAGS.num_layers, iterator)#train_set)
            with tf.variable_scope("model", reuse=True):
                print ("Creating dev model")
                model_dev = create_seq2seq_model(False, {'char': FLAGS.num_layers['char']},
                                                 dev_set.data_iter)
            with tf.variable_scope("model"):
                print ("Creating LM model")
                lm_model = create_lm_model(FLAGS.output_vocab_size["char"], FLAGS.hidden_size,
                                           data_iter=lm_set, scope="char")
            # These things need to be moved from original places so that optimizer used by
            # LM is properly initialized and stored

            model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
            if steps_done == 0:
                for var in tf.global_variables():
                    if "AdamLM" in var.name:
                        print var.name
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            else:
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
                tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
            # Prepare training data
            epoch = model.epoch.eval()
            epochs_left = FLAGS.max_epochs - epoch

            train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/train', tf.get_default_graph())
            asr_err_best = 1.0
            if steps_done > 0:
                # Some training has been done
                score_file = os.path.join(FLAGS.train_dir, "best.txt")
                # Check existence of such a file
                if os.path.isfile(score_file):
                    try:
                        asr_err_best = float(open(score_file).readline().strip("\n"))
                    except ValueError:
                        asr_err_best = 1

            print ("Best ASR error rate - %f" %asr_err_best)

            # This is the training loop.
            epc_time, ckpt_time, loss = 0.0, 0.0, 0.0
            current_step = 0
            lm_steps, lm_loss = 0, 0.0
            previous_errs = []

            # Run the LM initializer
            sess.run(lm_set.initialize_iterator())

            while epoch <= FLAGS.max_epochs:
                print("Epochs done: %d" %epoch)
                sys.stdout.flush()
                epc_start_time = time.time()
                ckpt_start_time = time.time()
                for iter_init in iter_init_list:
                    sess.run(iter_init)
                    # Track time taken
                    while True:
                        task = random.choice(["asr", "lm"])
                        #task = random.choice(["lm"])
                        if task == "lm":
                            try:
                                output_feed = [#lm_model.encoder_inputs, lm_model.seq_len,
                                               lm_model.updates, lm_model.losses]
                                _, lm_step_loss = sess.run(output_feed)
                                lm_loss += lm_step_loss/FLAGS.steps_per_checkpoint
                                lm_steps += 1
                                if lm_steps % FLAGS.steps_per_checkpoint == 0:
                                    #print (lm_enc_inp)
                                    #print (lm_seq_len)
                                    perplexity = math.exp(lm_loss) if lm_loss < 300 else float('inf')
                                    print ("LM steps: %d, Perplexity: %f" %(lm_steps, perplexity))
                                    lm_loss = 0.0
                            except tf.errors.OutOfRangeError:
                                # Run the LM initializer
                                sess.run(lm_set.initialize_iterator())
                        else:
                            try:
                                output_feed = [#model.decoder_inputs,
                                               model.updates,  model.losses]

                                #dec_inps, _, step_loss = sess.run(output_feed)
                                #print (dec_inps["char"].shape)
                                _, step_loss = sess.run(output_feed)
                                step_loss = step_loss["char"]

                                current_step += 1
                                loss += step_loss / FLAGS.steps_per_checkpoint

                                if current_step % FLAGS.steps_per_checkpoint == 0:
                                    # Print statistics for the previous epoch.
                                    perplexity = math.exp(loss) if loss < 300 else float('inf')
                                    ckpt_time = time.time() - ckpt_start_time

                                    print ("Step %d Learning rate %.4f Checkpoint time %.2f Perplexity "
                                           "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                                     ckpt_time, perplexity))

                                    decode_start_time = time.time()
                                    asr_err_cur = asr_decode(sess, model_dev, dev_set)
                                    decode_end_time = time.time() - decode_start_time

                                    print ("ASR error: %.4f, Decoding time: %s"
                                           %(asr_err_cur, timedelta(seconds=decode_end_time)))

                                    err_summary = get_summary(asr_err_cur, "ASR Error")
                                    train_writer.add_summary(err_summary, current_step)

                                    if (epoch >= FLAGS.min_epochs and asr_err_cur > max(previous_errs[-3:])):
                                        # Training has already happened for min epochs and the dev
                                        # error is getting worse w.r.t. the worst value in previous 3 checkpoints
                                        if model.learning_rate.eval() > 1e-4:
                                            sess.run(model.learning_rate_decay_op)
                                            print ("Learning rate decreased !!")
                                    previous_errs.append(loss)

                                    # Early stopping
                                    if asr_err_best > asr_err_cur:
                                        asr_err_best = asr_err_cur
                                        # Save model
                                        print("Best ASR Error rate: %.4f" % asr_err_best)
                                        print("Saving the best model !!")

                                        # Save the best score
                                        f = open(os.path.join(FLAGS.train_dir, "best.txt"), "w")
                                        f.write(str(asr_err_best))
                                        f.close()

                                        # Save the model in best model directory
                                        checkpoint_path = os.path.join(FLAGS.best_model_dir, "asr.ckpt")
                                        model_saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)

                                    # Also save the model for plotting
                                    checkpoint_path = os.path.join(FLAGS.train_dir, "asr.ckpt")
                                    model_saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)

                                    print ("\n")
                                    sys.stdout.flush()
                                    # Reinitialze tracking variables
                                    ckpt_start_time = time.time()
                                    loss = 0.0

                            except tf.errors.OutOfRangeError:
                                break

                print ("Total steps: %d" %model.global_step.eval())
                sess.run(model.epoch_incr)
                epoch += 1
                epc_time = time.time() - epc_start_time
                print ("\nEPOCH TIME: %s\n" %(str(timedelta(seconds=epc_time))))
                sys.stdout.flush()


def get_summary(value, tag):
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])


def reverse_swbd_normalizer():
    swbd_dict = {"!": "[laughter]",
                 "@": "[noise]",
                 "#": "[vocalized-noise]"}

    def normalizer(text):
        # Create a regex for match
        regex = re.compile("(%s)" % "|".join(map(re.escape, swbd_dict.keys())))
        # For each match, look-up corresponding value in dictionary
        return regex.sub(lambda match: swbd_dict[match.string[match.start() : match.end()]], text)

    return normalizer


def wp_array_to_sent(wp_array, reverse_char_vocab, normalizer):
    wp_id_list = list(wp_array)
    if data_utils.EOS_ID in wp_id_list:
        wp_id_list = wp_id_list[:wp_id_list.index(data_utils.EOS_ID)]
    wp_list = map(lambda piece_id:
                  tf.compat.as_str(reverse_char_vocab[piece_id]), wp_id_list)
    sent = (''.join(wp_list).replace('▁', ' ')).strip()
    return normalizer(sent)


def asr_decode(sess, model_dev, dev_set):
    # Load vocabularies.
    char_vocab_path = "/scratch2/asr_multi/data/lang/vocab/char.vocab"
    char_vocab, rev_char_vocab = data_utils.initialize_vocabulary(char_vocab_path)

    rev_normalizer = reverse_swbd_normalizer()

    gold_asr_file = os.path.join(FLAGS.train_dir, 'gold_asr.txt')
    decoded_asr_file = os.path.join(FLAGS.train_dir, 'decoded_asr.txt')
    raw_asr_file = os.path.join(FLAGS.train_dir, 'raw_asr.txt')

    fout_gold = open(gold_asr_file, 'w')
    fout_raw_asr = open(raw_asr_file, 'w')
    fout_asr = open(decoded_asr_file, 'w')

    total_errors = 0
    total_words = 0

    # Initialize the dev iterator
    sess.run(model_dev.data_iter.initializer)
    while True:
        try:
            output_feed = [model_dev.decoder_inputs["char"],
                           model_dev.outputs["char"]]

            gold_ids, output_logits = sess.run(output_feed)
            gold_ids = np.array(gold_ids[1:, :]).T
            batch_size = gold_ids.shape[0]

            outputs = np.argmax(output_logits, axis=1)
            outputs = np.reshape(outputs, (-1, batch_size))  # T*B

            to_decode = outputs.T  # B*T

            for sent_id in xrange(batch_size):
                gold_asr = wp_array_to_sent(gold_ids[sent_id, :], rev_char_vocab, rev_normalizer)
                decoded_asr = wp_array_to_sent(to_decode[sent_id, :], rev_char_vocab, rev_normalizer)
                raw_asr_words, decoded_words = data_utils.get_relevant_words(decoded_asr)
                _, gold_words = data_utils.get_relevant_words(gold_asr)

                total_errors += ed.eval(gold_words, decoded_words)
                total_words += len(gold_words)

                fout_gold.write('{}\n'.format(' '.join(gold_words)))
                fout_raw_asr.write('{}\n'.format(' '.join(raw_asr_words)))
                fout_asr.write('{}\n'.format(' '.join(decoded_words)))

        except tf.errors.OutOfRangeError:
            break

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
        pass

if __name__ == "__main__":
    parse_options()
    if FLAGS.eval_dev:
        decode()
    elif FLAGS.test:
        decode(test=True)
    else:
        train()

