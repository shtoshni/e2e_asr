from __future__ import absolute_import
from __future__ import division

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf
import cPickle as pickle
import argparse
import operator
from bunch import bunchify
import editdistance as ed
import glob

import data_utils
import seq2seq_model
import subprocess


tf.logging.set_verbosity(tf.logging.ERROR)


# Use the following buckets:
#_buckets = [(296, (85, 55)), (1498, (364, 221))]
#_buckets = [(1498, (364, 221))]
_buckets = [(210, (60, 50)), (346, (120, 110))]
bucket_size = [210, 346, 548, 850, 1500]
task_to_freq = {'phone':1.0, 'char':1.0}
eval_task_to_id = {'char':0}

NUM_THREADS = 1
PREFIX = '_short'
#PREFIX = ''
FLAGS = object()

def parse_tasks(task_string):
    task_dict = {'char': 0}
    task_list = list(set(task_string))
    if "p" in task_list:
        task_dict['phone'] = 1

    return task_dict


def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float, help="learning rate")
    parser.add_argument("-lr_decay", "--learning_rate_decay_factor", default=0.95, type=float, help="multiplicative decay factor for learning rate")
    parser.add_argument("-opt", "--optimizer", default="adam", type=str, help="Optimizer")
    parser.add_argument("-bsize", "--batch_size", default=64, type=int, help="Mini-batch Size")

    parser.add_argument("-esize", "--embedding_size", default=256, type=int, help="Embedding Size")
    parser.add_argument("-hsize", "--hidden_size", default=256, type=int, help="Hidden layer size")
    parser.add_argument("-hsize_decoder", "--hidden_size_decoder", default=256, type=int, help="Hidden layer size")

    parser.add_argument("-num_layers_phone", "--num_layers_phone", default=1, type=int, help="Number of layers to decode side")
    parser.add_argument("-num_layers_char", "--num_layers_char", default=1, type=int, help="Number of layers to decode side")
    parser.add_argument("-num_layers_phone_decoder", "--num_layers_phone_decoder", default=1, type=int, help="Number of layers to decode side")
    parser.add_argument("-num_layers_char_decoder", "--num_layers_char_decoder", default=1, type=int, help="Number of layers to decode side")
    parser.add_argument("-max_gnorm", "--max_gradient_norm", default=5.0, type=float, help="Maximum allowed norm of gradients")

    parser.add_argument("-tv_file_phone", "--target_vocab_file_phone", default="phone.txt", type=str, help="Vocab file for phone target")
    parser.add_argument("-tv_file_char", "--target_vocab_file_char", default="char.txt", type=str, help="Vocab file for character target")

    parser.add_argument("-vocab_dir", "--vocab_dir", default="/share/data/speech/shtoshni/research/asr_multi/data/vocab", type=str, help="Vocab directory")
    #parser.add_argument("-data_dir", "--data_dir", default="/share/data/speech/shtoshni/research/asr_multi/data", type=str, help="Data directory")
    #parser.add_argument("-data_dir", "--data_dir", default="/share/data/speech/shtoshni/research/asr_multi/data/queue_delta", type=str, help="Data directory")
    parser.add_argument("-data_dir", "--data_dir", default="/scratch/asr_multi/data/ctc_data", type=str, help="Data directory")
    #parser.add_argument("-data_dir", "--data_dir", default="/scratch/asr_multi/data", type=str, help="Data directory")
    parser.add_argument("-tb_dir", "--train_base_dir", default="/share/data/speech/shtoshni/research/asr_multi/models", type=str, help="Training directory")
    parser.add_argument("-tasks", "--tasks", default="", type=str, help="Auxiliary task choices")

    parser.add_argument("-bi_dir", "--bi_dir", default=False, action="store_true", help="Make encoder bi-directional")
    parser.add_argument("-skip_step", "--skip_step", default=1, type=int, help="Frame skipping factor as we go up the stacked layers")

    parser.add_argument("-lstm", "--lstm", default=True, action="store_true", help="RNN cell to use")
    parser.add_argument("-out_prob", "--output_keep_prob", default=1.0, type=float, help="Output keep probability for dropout")
    parser.add_argument("-apply_dropout", "--apply_dropout", default=False, action="store_true", help="Use dropout or not")

    ## Additional features
    parser.add_argument("-use_conv", "--use_convolution", default=False, action="store_true", help="Use convolution feature in attention")
    parser.add_argument("-conv_filter", "--conv_filter_dimension", default=80, type=int, help="Convolution filter width dimension")
    parser.add_argument("-conv_channel", "--conv_num_channel", default=3, type=int, help="Number of channels in the convolution feature extracted")

    parser.add_argument("-feat_len", "--feat_length", default=80, type=int, help="Number of features per frame")
    parser.add_argument("-base_pyramid", "--base_pyramid", default=False, action="store_true", help="Do pyramid at feature level as well?")
    parser.add_argument("-var_noise", "--var_noise", default=False, action="store_true", help="Add variational noise?")
    parser.add_argument("-sch_samp", "--sch_samp", default=False, action="store_true", help="Do pyramid at feature level as well?")
    parser.add_argument("-l2_weight", "--l2_weight", default=0.0, type=float, help="L2 loss weight")

    parser.add_argument("-num_files", "--num_files", default=50, type=int, help="Num files")
    parser.add_argument("-max_epochs", "--max_epochs", default=30, type=int, help="Max epochs")
    parser.add_argument("-num_check", "--steps_per_checkpoint", default=100, type=int, help="Number of steps before updated model is saved")
    parser.add_argument("-eval", "--eval_dev", default=False, action="store_true", help="Get dev set results using the last saved model")
    parser.add_argument("-test", "--test", default=False, action="store_true", help="Get test results using the last saved model")
    parser.add_argument("-run_id", "--run_id", default=0, type=int, help="Run ID")

    args = parser.parse_args()
    arg_dict = vars(args)

    data_limits = {}
    data_limits['FEAT_LEN'] = arg_dict['feat_length']
    data_limits['_PAD_VEC'] = np.zeros(data_limits['FEAT_LEN'], dtype=np.float32)
    data_limits = bunchify(data_limits)

    arg_dict['task_to_id'] = parse_tasks(arg_dict['tasks'])

    feat_length_string = ""
    if data_limits.FEAT_LEN != 80:
        feat_length_string = "fl_" + str(data_limits.FEAT_LEN) + "_"

    skip_string = ""
    if arg_dict['skip_step'] != 1:
        skip_string = "skip_" + str(arg_dict['skip_step']) + "_"

    bi_dir_string = ""
    if arg_dict['bi_dir'] != False:
        bi_dir_string = "bi_dir_"

    base_pyramid_string = ""
    if arg_dict['base_pyramid'] != False:
        base_pyramid_string = "bp_"

    conv_string = ""
    if arg_dict['use_convolution']:
        conv_string = "use_conv_"
        conv_string += "filter_dim_" + str(arg_dict['conv_filter_dimension']) + "_"
        conv_string += "num_channel_" + str(arg_dict['conv_num_channel']) + "_"

    num_layer_string = ""
    for task in arg_dict['task_to_id']:
        num_layer_string += 'nl' + task + '_' + str(arg_dict['num_layers_' + task]) + '_'


    train_dir = ('lr_' + str(arg_dict['learning_rate']) + '_' +
                'bsize_' + str(arg_dict['batch_size']) + '_' +
                'esize_' + str(arg_dict['embedding_size']) + '_' +
                'hsize_' + str(arg_dict['hidden_size']) + '_' +
                'hsize_dec_' + str(arg_dict['hidden_size_decoder']) + '_' +

                skip_string +
                bi_dir_string +
                base_pyramid_string +
                conv_string +

                num_layer_string +
                feat_length_string +

                'out_prob_' + str(arg_dict['output_keep_prob']) + '_' +
                'run_id_' + str(arg_dict['run_id']) + '_' +
                'cmvn')


    arg_dict['train_dir'] = os.path.join(arg_dict['train_base_dir'], train_dir)
    arg_dict['apply_dropout'] = False

    arg_dict['num_layers'] = {}
    for task in arg_dict['task_to_id']:
        arg_dict['num_layers'][task] = arg_dict['num_layers_' + task]

    arg_dict['num_layers_decoder'] = {}
    for task in arg_dict['task_to_id']:
        arg_dict['num_layers_decoder'][task] = arg_dict['num_layers_' + task + '_decoder']

    arg_dict['target_vocab_file'] = {}
    for task in arg_dict['task_to_id']:
        arg_dict['target_vocab_file'][task] = arg_dict['target_vocab_file_' + task]

    arg_dict['output_vocab_size'] = {}
    for task, task_id in arg_dict['task_to_id'].iteritems():
        target_vocab, _ = data_utils.initialize_vocabulary(os.path.join(arg_dict['vocab_dir'], \
                arg_dict['target_vocab_file'][task]))
        arg_dict['output_vocab_size'][task] = len(target_vocab)

    if not arg_dict['test'] and not arg_dict['eval_dev']:
        arg_dict['apply_dropout'] = True
        if not os.path.exists(arg_dict['train_dir']):
            os.makedirs(arg_dict['train_dir'])

        ## Sort the arg_dict to create a parameter file
        parameter_file = 'parameters.txt'
        sorted_args = sorted(arg_dict.items(), key=operator.itemgetter(0))

        with open(os.path.join(arg_dict['train_dir'], parameter_file), 'w') as g:
            for arg, arg_val in sorted_args:
                sys.stdout.write(arg + "\t" + str(arg_val) + "\n")
                sys.stdout.flush()
                g.write(arg + "\t" + str(arg_val) + "\n")

    options = bunchify(arg_dict)
    options.data_limits = data_limits
    return options

def load_dev_data():
    dev_data_path = os.path.join(FLAGS.data_dir, 'dev' + PREFIX + '.pickle')
    dev_set = pickle.load(open(dev_data_path))

    num_dev_samples = sum([len(bucket_instances) for bucket_instances in dev_set])
    print ("Number of dev samples: %d" %num_dev_samples)

    return dev_set


def get_train_data():
    #train_data_path = os.path.join(FLAGS.data_dir, "train" + "0")#".?")
    #all_files = glob.glob(FLAGS.data_dir + "/train*")
    all_files = glob.glob(FLAGS.data_dir + "/train0.1*")
    train_files = []
    for file_name in all_files:
        batch_idx = int(file_name.split(".")[1])
        if batch_idx <= 40:
            train_files.append(file_name)
    train_files.sort()
    train_files = train_files[:FLAGS.num_files]
    print (train_files)
    print (len(train_files))
    number_of_batches = 0
    for train_file in train_files:
        print(train_file)
        number_of_instances = sum([1 for _ in tf.python_io.tf_record_iterator(train_file)])
        ## Using ceil below since we allow for smaller final batch
        number_of_batches += int(np.ceil(number_of_instances/float(FLAGS.batch_size)))

    return train_files, number_of_batches


def get_model_graph(session, forward_only, task_to_id=None, queue=None):
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.output_vocab_size, _buckets,
      FLAGS.hidden_size, FLAGS.hidden_size_decoder,
      FLAGS.num_layers, FLAGS.num_layers_decoder,
      FLAGS.embedding_size, FLAGS.skip_step, FLAGS.bi_dir,
      FLAGS.use_convolution, FLAGS.conv_filter_dimension, FLAGS.conv_num_channel,
      FLAGS.max_gradient_norm, FLAGS.batch_size, FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor, FLAGS.optimizer, FLAGS.data_limits,
      queue=queue,
      use_lstm=FLAGS.lstm, output_keep_prob=FLAGS.output_keep_prob,
      forward_only=forward_only,
      base_pyramid=FLAGS.base_pyramid,
      l2_weight=FLAGS.l2_weight,
      sch_samp=FLAGS.sch_samp,
      task_to_id=task_to_id,
      apply_dropout=FLAGS.apply_dropout)

  return model

def create_model(session, forward_only, model_path=None, task_to_id=None, queue=None):
  """Create translation model and initialize or load parameters in session."""
  model = get_model_graph(session, forward_only, task_to_id=task_to_id, queue=queue)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path) and not model_path:
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
      model, steps_done = create_model(sess, forward_only=False, task_to_id=FLAGS.task_to_id, queue=bucket_queue)
    with tf.variable_scope("model", reuse=True):
        model_dev = get_model_graph(sess, forward_only=True, task_to_id=eval_task_to_id)


    # Prepare training data
    epoch = model.epoch.eval()
    epochs_left = FLAGS.max_epochs - epoch

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Test dev results
    dev_set = load_dev_data()
    asr_err_best = 1#asr_decode(model_dev, sess, dev_set)

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    bucket_counter = [0] * len(bucket_size)

    total_minibatches = num_batch

    while epoch <= FLAGS.max_epochs:
        try:
            while not coord.should_stop():
                print("Epochs done: %d" %epoch)
                sys.stdout.flush()

                for i in xrange(total_minibatches):
                    task = "char"
                    cur_task = "char"
                    start_time = time.time()
                    output_feed = [model.seq_len,
                                    model.updates,  # Update Op that does SGD.
                                    model.gradient_norms,  # Gradient norm.
                                    model.losses]  # Loss for this batch.

                    seq_len, _, grad_norm, step_loss = sess.run(output_feed)
                    example_len = seq_len[0]
                    minibatch_bucket_id = data_utils.get_bucket_id(bucket_size, example_len)
                    bucket_counter[minibatch_bucket_id] += 1


                    step_loss = step_loss[task]

                    step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                    sys.exit()
                    if cur_task == 'char':
                        #print(inp.shape)
                        #print (out["char"].shape)
                        #print (seq_len.shape)
                        #print (np.max(seq_len_target["char"]))
                        loss += step_loss / FLAGS.steps_per_checkpoint
                        current_step += 1

                        # Once in a while, we save checkpoint, print statistics, and run evals.
                        if current_step % FLAGS.steps_per_checkpoint == 0:
                          # Print statistics for the previous epoch.
                          perplexity = math.exp(loss) if loss < 300 else float('inf')
                          for bucket_id in xrange(len(bucket_counter)):
                              print ("Bucket ID: %d, Counter: %d" %(bucket_id, bucket_counter[bucket_id]))

                          print ("global step %d learning rate %.4f step-time %.2f perplexity "
                               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                         step_time, perplexity))

                          print("Gradient Norm %.4f"%(grad_norm))
                          # Decrease learning rate if no improvement was seen over last 3 times.
                          if len(previous_losses) > 0 and loss > previous_losses[-1]:
                            sess.run(model.learning_rate_decay_op)
                          previous_losses.append(loss)

                          asr_err_cur = asr_decode(model_dev, sess, dev_set)
                          print ("ASR error: %.4f" %(asr_err_cur))
                          ## Early stopping - ONLY UPDATING MODEL IF BETTER PERFORMANCE ON DEV
                          if asr_err_best > asr_err_cur:
                            asr_err_best = asr_err_cur
                            # Save model
                            print("Best ASR Error rate: %.4f" % asr_err_best)
                            print("Saving updated model")
                            sys.stdout.flush()
                            checkpoint_path = os.path.join(FLAGS.train_dir, "asr_small.ckpt")
                            model.saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)
                          step_time, loss = 0.0, 0.0

                ## Update epoch counter
                sess.run(model.epoch_incr)
                epoch += 1

        except Exception as e:
          coord.request_stop(e)
          coord.join(threads)
          break

    coord.request_stop()
    coord.join(threads)


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
  for bucket_id in xrange(len(_buckets)):
    bucket_size = len(dev_set[bucket_id])
    offsets = np.arange(0, bucket_size, batch_size)
    for batch_offset in offsets:
        all_examples = dev_set[bucket_id][batch_offset:batch_offset+batch_size]

        model_dev.batch_size = len(all_examples)
        log_mels = [x[0] for x in all_examples]
        gold_ids = [x[1] for x in all_examples]
        sent_id_vals = [x[2] for x in all_examples]
        dec_ids = [[]] * len(gold_ids)

        encoder_inputs, decoder_inputs,  seq_len, seq_len_target = \
                model_dev.get_batch({bucket_id: zip(log_mels, gold_ids)}, bucket_id, task='char', do_eval=True)


        _, _, output_logits = model_dev.step(sess, encoder_inputs, decoder_inputs,\
                         seq_len, seq_len_target, False)

        #print (output_logits.shape)
        outputs = np.argmax(output_logits, axis=1)
        outputs = np.reshape(outputs, (max(seq_len_target['char']), model_dev.batch_size)) ##T*B

        to_decode = np.array(outputs).T ## T * B and the transpose makes it B*T

        num_dev_sents += to_decode.shape[0]
        for sent_id in range(to_decode.shape[0]):
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
          fout_raw_asr.write('{}\n'.format(' '.join(raw_asr_words)))
          fout_asr.write('{}\n'.format(' '.join(decoded_words)))

  # Write to file
  fout_gold.close()
  fout_raw_asr.close()
  fout_asr.close()

  return float(total_errors)/float(total_words)


def dump_trainable_vars():
    model_prefix = FLAGS.train_dir.split("/")[-1]
    model_file = os.path.join(FLAGS.train_dir, "s2p_tuned_" + model_prefix + ".pickle")

    with open(model_file, "w") as f:
        var_name_to_val = {}
        for var in tf.trainable_variables():
            var_name_to_val[var.name] = var.eval()

        pickle.dump(var_name_to_val, f)


def decode():
  """ Decode file sentence-by-sentence  """
  with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)) as sess:
    # Create model and load parameters.
    with tf.variable_scope("model"):
      model_dev, steps_done = create_model(sess, forward_only=True, task_to_id=eval_task_to_id)

    print ("Epochs done: %d" %model_dev.epoch.eval())
    dump_trainable_vars()
    dev_set = load_dev_data()

    start_time = time.time()
    asr_decode(model_dev, sess, dev_set)
    time_elapsed = time.time() - start_time
    print("Decoding all dev time: ", time_elapsed)



if __name__ == "__main__":
    FLAGS = parse_options()
    if FLAGS.test:
        self_test()
    elif FLAGS.eval_dev:
        decode()
    else:
        train()

