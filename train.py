from __future__ import absolute_import
from __future__ import division

import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

from os import path
import copy
import random
import sys
import time

import argparse
import operator
import glob
import re
from datetime import timedelta

from bunch import Bunch
import tensorflow as tf

import data_utils
import tf_utils
import numpy as np
from attn_decoder import AttnDecoder
from encoder import Encoder
from lm_encoder import LMEncoder
from lm_model import LMModel
from seq2seq_model import Seq2SeqModel
from speech_dataset import SpeechDataset
from lm_dataset import LMDataset
from base_params import BaseParams
from eval_model import Eval


class Train(BaseParams):

    @classmethod
    def class_params(cls):
        params = Bunch()

        params['batch_size'] = 128
        params['buck_batch_size'] = [128, 128, 64, 64, 32]
        #params['buck_batch_size'] = [32, 32, 16, 16, 8]
        params['max_steps'] = 15000
        params['min_steps'] = 6000
        params['max_epochs'] = 10
        params['min_lr_rate'] = 1e-4
        params['feat_length'] = 80

        # Data directories
        params['data_dir'] = "/scratch/asr_multi/data/tfrecords"
        params['lm_data_dir'] = "/scratch/asr_multi/data/tfrecords/fisher/red_0.7"
        params['vocab_dir'] = "/share/data/speech/shtoshni/research/datasets/asr_swbd/lang/vocab"

        params['train_base_dir'] = "/scratch/asr_multi/models"
        # The train_dir and best_model_dir are supplied by the process_args() in main.py
        params['train_dir'] = "/scratch"
        params['best_model_dir'] = "/scratch"

        params['lm_prob'] = 0.0
        params['lm_params'] = LMModel.class_params()
        params['lm_enc_params'] = LMEncoder.class_params()

        params['run_id'] = 1
        params['steps_per_checkpoint'] = 500

        # Pretrained models path
        params["pretrain_lm_path"] = ""
        # Pretrain one layer model path
        params["pretrain_ol_path"] = ""

        params["chaos"] = False
        params["subset_file"] = ""
        return params

    def __init__(self, model_params, train_params=None):
        if train_params is None:
            self.params = self.class_params()
        else:
            self.params = train_params
        params = self.params

        self.seq2seq_params = model_params
        self.eval_model = None

    def load_train_subset_file(self, subset_file):
        subset_file_dict = {}
        try:
            with open(subset_file) as f:
                for line in f.readlines():
                    subset_file_dict[line.strip()] = 0
        except Error:
            subset_file_dict = {}
        return subset_file_dict

    def get_data_sets(self, logging=True):
        params = self.params
        buck_train_sets = []
        total_train_files = 0

        dataset_params_def = Bunch()
        dataset_params_def.batch_size = params.batch_size
        dataset_params_def.feat_length = params.feat_length

        if params.subset_file:
            subset_file_dict = self.load_train_subset_file(params.subset_file)
        else:
            subset_file_dict = None

        for batch_id, batch_size in enumerate(params.buck_batch_size):
            dataset_params = copy.deepcopy(dataset_params_def)
            dataset_params.batch_size = batch_size

            buck_train_files = glob.glob(path.join(
                params.data_dir, "train_1k." + str(batch_id) + ".*"))
            if subset_file_dict:
                buck_train_files = [train_file for train_file in buck_train_files if path.basename(train_file) in subset_file_dict ]
            random.shuffle(buck_train_files)
            total_train_files += len(buck_train_files)
            buck_train_set = SpeechDataset(dataset_params, buck_train_files, isTraining=True)
            buck_train_sets.append(buck_train_set)
        if logging:
            print ("Total train files: %d" %total_train_files)

        # Dev set
        dev_files = glob.glob(path.join(params.data_dir, "dev*"))
        if logging:
            print ("Total dev files: %d" %len(dev_files))
        dev_set = SpeechDataset(dataset_params_def, dev_files,
                                isTraining=False)
        return buck_train_sets, dev_set


    def get_lm_files(self):
        params = self.params
        lm_files = glob.glob(path.join(params.lm_data_dir, "lm*"))
        return lm_files


    def restore_lm_variables(self, sess, lm_ckpt_path):
        """Restore LM variables."""
        ckpt_reader = tf.train.NewCheckpointReader(lm_ckpt_path)
        ckpt_vars = ckpt_reader.get_variable_to_shape_map()

        var_names_map = {var.op.name: var for var in tf.trainable_variables()}

        lm_lstm_w = None
        lm_lstm_b = None
        for var_name, var in var_names_map.items():
            if "lstm" in var_name:
                # We just have to handle lstm cells differently so as not to confuse
                # with decoder cell
                if "rnn/lm" in var_name:
                    if "kernel" in var_name:
                        lm_lstm_w = var
                    else:
                        lm_lstm_b = var
                continue
            else:
                if var_name in ckpt_vars:
                    try:
                        sess.run(var.assign(ckpt_reader.get_tensor(var_name)))
                        print ("Using pre-trained: %s" %var.name)
                    except ValueError:
                        print ("Shape wanted: %s, Shape stored: %s for %s"
                               %(str(var.shape), str(ckpt_reader.get_tensor(var_name).shape),
                                 var_name))

        print ("Using pretrained model/rnn_decoder_char/rnn/lm/basic_lstm_cell/kernel")
        print ("Using pretrained model/rnn_decoder_char/rnn/lm/basic_lstm_cell/bias")
        sess.run(lm_lstm_w.assign(ckpt_reader.get_tensor(
            "model/rnn_decoder_char/rnn/basic_lstm_cell/kernel")))
        sess.run(lm_lstm_b.assign(ckpt_reader.get_tensor(
            "model/rnn_decoder_char/rnn/basic_lstm_cell/bias")))


    def restore_ol_variables(self, sess, ol_ckpt_path):
        """Restore LM variables."""
        ckpt_reader = tf.train.NewCheckpointReader(ol_ckpt_path)
        ckpt_vars = ckpt_reader.get_variable_to_shape_map()

        var_names_map = {var.op.name: var for var in tf.trainable_variables()}

        for var_name, var in var_names_map.items():
            if "embedding" in var_name:
                # Restore embedding from LM
                continue
            else:
                if var_name in ckpt_vars:
                    try:
                        sess.run(var.assign(ckpt_reader.get_tensor(var_name)))
                        print ("Using pre-trained: %s" %var.name)
                    except ValueError:
                        print ("Shape wanted: %s, Shape stored: %s for %s"
                               %(str(var.shape), str(ckpt_reader.get_tensor(var_name).shape),
                                 var_name))


    def create_eval_model(self, dev_set, standalone=False):
        with tf.variable_scope("model", reuse=(True if not standalone else None)):
            print ("Creating dev model")
            dev_seq2seq_params = copy.deepcopy(self.seq2seq_params)
            dev_seq2seq_params.tasks = {'char'}
            dev_seq2seq_params.num_layers = {'char': dev_seq2seq_params.num_layers['char']}
            model_dev = Seq2SeqModel(dev_set.data_iter, isTraining=False,
                                     params=dev_seq2seq_params)

            params = Bunch()
            params.best_model_dir = self.params.best_model_dir
            params.vocab_dir = self.params.vocab_dir

            self.eval_model = Eval(model_dev, params=params)

    @staticmethod
    def check_progess(previous_errs, num=10):
        if len(previous_errs) > num:
            if min(previous_errs) != min(previous_errs[-num:]):
                return False
        return True

    def train(self):
        """Train a sequence to sequence speech recognizer!"""
        params = self.params
        model_params = self.seq2seq_params

        with tf.Graph().as_default():
            # Set the random seeds
            if not params.chaos:
                # Random seeds controlled
                tf.set_random_seed(10)
                random.seed(10)
            else:
                # For 4 hr GPU cycles introducing randomness is good
                tf.set_random_seed(int(time.time()))
                random.seed(int(time.time()))

            # Bucket train sets
            buck_train_sets, dev_set = self.get_data_sets()
            with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1)) as sess:
                handle = tf.placeholder(tf.string, shape=[])
                iterator = tf.data.Iterator.from_string_handle(
                    handle, buck_train_sets[0].data_set.output_types,
                    buck_train_sets[0].data_set.output_shapes)

                with tf.variable_scope("model", reuse=None):
                    model = Seq2SeqModel(iterator, True, model_params)
                    # Create eval model

                self.create_eval_model(dev_set)

                if params.lm_prob > 0:
                    # Create LM dataset
                    lm_files = self.get_lm_files()

                    # Create LM model
                    with tf.variable_scope("model", reuse=None):
                        print ("Creating LM model")
                        sys.stdout.flush()
                        lm_model = LMModel(LMEncoder(params=params.lm_enc_params),
                                           data_files=lm_files,
                                           params=params.lm_params)

                model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
                best_model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

                ckpt = tf.train.get_checkpoint_state(params.train_dir)
                if not ckpt:
                    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                    if params.pretrain_lm_path:
                        #tf_utils.restore_common_variables(sess, params.pretrain_lm_path)
                        self.restore_lm_variables(sess, params.pretrain_lm_path)
                    if params.pretrain_ol_path:
                        self.restore_ol_variables(sess, params.pretrain_ol_path)


                else:
                    tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
                # Prepare training data
                epoch = model.global_step.eval()/3006  # For default setup it's 3006

                train_writer = tf.summary.FileWriter(params.train_dir +
                                                     '/summary', tf.get_default_graph())
                asr_err_best = 1.0
                if ckpt:
                    # Some training has been done
                    score_file = os.path.join(params.train_dir, "best.txt")
                    # Check existence of such a file
                    if os.path.isfile(score_file):
                        try:
                            asr_err_best = float(open(score_file).readline().strip("\n"))
                        except ValueError:
                            pass

                print ("\nBest ASR error rate - %f" %asr_err_best)
                sys.stdout.flush()

                # This is the training loop.
                epc_time, loss = 0.0, 0.0
                ckpt_start_time = time.time()
                current_step = 0
                # Last time step when LR was decreased
                last_lr_decrease_step = 0

                if params.lm_prob > 0:
                    lm_steps, lm_loss = 0, 0.0
                    sess.run(lm_model.data_iter.initializer)
                try:
                    with open(path.join(params.train_dir, "last_lr_dec.txt"), "r") as err_f:
                        last_lr_decrease_step = int(err_f.readline().strip())
                except IOError:
                    pass


                while epoch <= params.max_epochs:
                    print("\nEpochs done: %d" %epoch)
                    sys.stdout.flush()
                    epc_start_time = time.time()

                    active_handle_list = []
                    for train_set in buck_train_sets:
                        sess.run(train_set.data_iter.initializer)
                        active_handle_list.append(sess.run(train_set.data_iter.string_handle()))

                    handle_idx_dict = dict(zip(active_handle_list, list(range(len(active_handle_list)))))


                    while True:
                        task = ("lm" if (params.lm_prob > random.random()) else "asr")
                        if task == "lm":
                            try:
                                output_feed = [lm_model.updates, lm_model.losses]
                                _, lm_step_loss = sess.run(output_feed)
                                lm_loss += lm_step_loss/params.steps_per_checkpoint
                                lm_steps += 1
                                if lm_steps % params.steps_per_checkpoint == 0:
                                    perplexity = math.exp(lm_loss) if lm_loss < 300 else float('inf')
                                    print ("LM steps: %d, Perplexity: %f" %(
                                        lm_model.lm_global_step.eval(), perplexity))
                                    sys.stdout.flush()

                                    lm_summary = tf_utils.get_summary(perplexity, "LM Perplexity")
                                    train_writer.add_summary(lm_summary, model.global_step.eval())

                                    lm_loss = 0.0
                            except tf.errors.OutOfRangeError:
                                # Create LM dataset again - Another shuffle
                                lm_model.update_iterator()
                                sess.run(lm_model.epoch_incr)
                                sess.run(lm_model.data_iter.initializer)
                                print ("LM Epoch done %d !!" %lm_model.epoch.eval())

                        else:
                            # Pick the handle for the smallest utterances
                            cur_handle = active_handle_list[0]
                            try:
                                output_feed = [model.updates, model.losses]

                                _, step_loss = sess.run(output_feed, feed_dict={handle: cur_handle})
                                step_loss = step_loss["char"]

                                current_step += 1
                                loss += step_loss / params.steps_per_checkpoint

                                cur_global_step = model.global_step.eval()
                                if cur_global_step > params.max_steps:
                                    break
                                if (cur_global_step >= params.min_steps):
                                    if cur_global_step >= (last_lr_decrease_step + 3000):
                                        # Roughly an epoch after last decrease
                                        # TODO: Change the constant 3K to actual epoch size
                                        sess.run(model.learning_rate_decay_op)
                                        print ("Learning rate decreased !!")
                                        print ("LR: %.4f" % model.learning_rate.eval())

                                        # Update last LR decrease step and write it in file
                                        last_lr_decrease_step = cur_global_step
                                        with open(path.join(params.train_dir,
                                                            "last_lr_dec.txt"), "w") as err_f:
                                            err_f.write(str(last_lr_decrease_step))
                                            err_f.flush()

                                        sys.stdout.flush()

                                if current_step % params.steps_per_checkpoint == 0:
                                    # Print statistics for the previous epoch.
                                    perplexity = math.exp(loss) if loss < 300 else float('inf')
                                    ckpt_time = time.time() - ckpt_start_time

                                    print ("Step %d Learning rate %.4f Checkpoint time %.2f Perplexity "
                                           "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                                     ckpt_time, perplexity))
                                    sys.stdout.flush()

                                    loss_summary = tf_utils.get_summary(perplexity, "ASR Perplexity")
                                    train_writer.add_summary(loss_summary, model.global_step.eval())

                                    lr_summary = tf_utils.get_summary(model.learning_rate.eval(), "Learning rate")
                                    train_writer.add_summary(lr_summary, model.global_step.eval())

                                    decode_start_time = time.time()
                                    asr_err_cur = self.eval_model.greedy_decode(sess)
                                    decode_end_time = time.time() - decode_start_time

                                    print ("ASR error: %.4f, Decoding time: %s"
                                           %(asr_err_cur, timedelta(seconds=decode_end_time)))
                                    sys.stdout.flush()
                                    with open(path.join(params.train_dir, "asr_err.txt"), "a") as err_f:
                                        err_f.write(str(asr_err_cur) + "\n")

                                    err_summary = tf_utils.get_summary(asr_err_cur, "ASR Error")
                                    train_writer.add_summary(err_summary, model.global_step.eval())

                                    # Early stopping
                                    if asr_err_best > asr_err_cur:
                                        asr_err_best = asr_err_cur
                                        # Save model
                                        print("Best ASR Error rate: %.4f" % asr_err_best)
                                        print("Saving the best model !!")
                                        sys.stdout.flush()

                                        # Save the best score
                                        with open(os.path.join(params.train_dir, "best.txt"), "w") as asr_f:
                                            asr_f.write(str(asr_err_best))

                                        # Save the model in best model directory
                                        checkpoint_path = os.path.join(params.best_model_dir, "asr.ckpt")
                                        best_model_saver.save(sess, checkpoint_path,
                                                              global_step=model.global_step,
                                                              write_meta_graph=False)

                                    # Also save the model for plotting
                                    checkpoint_path = os.path.join(params.train_dir, "asr.ckpt")
                                    model_saver.save(sess, checkpoint_path, global_step=model.global_step,
                                                     write_meta_graph=False)

                                    print ("\n")
                                    sys.stdout.flush()
                                    # Reinitialze tracking variables
                                    ckpt_start_time = time.time()
                                    loss = 0.0

                            except tf.errors.OutOfRangeError:
                                # 0 out the prob of the given handle
                                del active_handle_list[0]
                                if not active_handle_list:
                                    break


                    cur_global_step = model.global_step.eval()
                    print ("Total steps: %d" %cur_global_step)
                    if cur_global_step > params.max_steps:
                        break
                    sess.run(model.epoch_incr)
                    epoch += 1
                    epc_time = time.time() - epc_start_time
                    print ("\nEPOCH TIME: %s\n" %(str(timedelta(seconds=epc_time))))
                    sys.stdout.flush()


                    print ("Reshuffling ASR training data!")
                    buck_train_sets, dev_set = self.get_data_sets(logging=False)


    @classmethod
    def add_parse_options(cls, parser):
        # Training params
        parser.add_argument("-lm_prob", default=0.0, type=float,
                            help="Prob. of running the LM task")
        parser.add_argument("-run_id", "--run_id", default=0, type=int, help="Run ID")
        parser.add_argument("-data_dir", default="/scratch/asr_multi/data/tfrecords",
                            type=str, help="Data directory")
        parser.add_argument("-lm_data_dir",
                            default="/scratch/asr_multi/data/tfrecords/lm_all",
                            type=str, help="Data directory")
        parser.add_argument("-vocab_dir", "--vocab_dir", default="/share/data/speech/"
                            "shtoshni/research/datasets/asr_swbd/lang/vocab",
                            type=str, help="Vocab directory")
        parser.add_argument("-tb_dir", "--train_base_dir",
                            default="/scratch/asr_multi/models",
                            type=str, help="Training directory")
        parser.add_argument("-feat_len", "--feat_length", default=80, type=int,
                            help="Number of features per frame")
        parser.add_argument("-steps_per_checkpoint", default=500,
                            type=int, help="Gradient steps per checkpoint")
        parser.add_argument("-min_steps", "--min_steps", default=6000, type=int,
                            help="min steps before decreasing learning rate")
        parser.add_argument("-max_steps", default=15000, type=int,
                            help="max training steps")

        parser.add_argument("-pretrain_lm_path",
                            default="/share/data/speech/shtoshni/research/asr_multi/"
                            "code/lm/models/best_models/run_id_301/lm.ckpt-250000",
                            type=str, help="Pretrain language model path")
        parser.add_argument("-pretrain_ol_path",
                            default="/share/data/speech/shtoshni/research/asr_multi/models/best_models/"
                            "skip_2_lstm_lm_prob_0.0_run_id_2702/asr.ckpt-28500", type=str,
                            help="Pretrain phone model path")
        parser.add_argument("-chaos", default=False, action="store_true",
                            help="Random seed is not controlled if set")
        parser.add_argument("-subset_file", default="", type=str,
                            help="Subset file")

