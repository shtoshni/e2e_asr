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
        params['lm_buck_batch_size'] = [256, 256, 128, 128, 64]
        params['buck_batch_size'] = [128, 128, 64, 64, 32]
        params['max_epochs'] = 30
        params['min_steps'] = 25000
        params['feat_length'] = 80

        # Data directories
        params['data_dir'] = "/scratch2/asr_multi/data/tfrecords"
        params['lm_data_dir'] = "/scratch2/asr_multi/data/tfrecords/fisher/red_0.7"
        params['vocab_dir'] = "/share/data/speech/shtoshni/research/datasets/asr_swbd/lang/vocab"

        params['train_base_dir'] = "/scratch2/asr_multi/models"
        # The train_dir and best_model_dir are supplied by the process_args() in main.py
        params['train_dir'] = "/scratch"
        params['best_model_dir'] = "/scratch"

        params['lm_prob'] = 0.0  #0.5
        params['lm_params'] = LMModel.class_params()

        params['run_id'] = 1
        params['steps_per_checkpoint'] = 500

        # Pretrained models path
        params["pretrain_lm_path"] = ""
        params["pretrain_phone_path"] = ""

        params["chaos"] = False
        return params

    def __init__(self, model_params, train_params=None):
        if train_params is None:
            self.params = self.class_params()
        else:
            self.params = train_params
        params = self.params

        self.seq2seq_params = model_params
        self.eval_model = None

    def get_data_sets(self):
        params = self.params
        buck_train_sets = []
        total_train_files = 0

        dataset_params_def = Bunch()
        dataset_params_def.batch_size = params.batch_size
        dataset_params_def.feat_length = params.feat_length

        for batch_id, batch_size in enumerate(params.buck_batch_size):
            dataset_params = copy.deepcopy(dataset_params_def)
            dataset_params.batch_size = batch_size
            buck_train_files = glob.glob(path.join(
                params.data_dir, "train_1k." + str(batch_id) + ".*"))
            total_train_files += len(buck_train_files)
            buck_train_set = SpeechDataset(dataset_params, buck_train_files, isTraining=True)
            buck_train_sets.append(buck_train_set)
        print ("Total train files: %d" %total_train_files)

        # Dev set
        dev_files = glob.glob(path.join(params.data_dir, "dev*"))
        print ("Total dev files: %d" %len(dev_files))
        dev_set = SpeechDataset(dataset_params_def, dev_files,
                                isTraining=False)
        return buck_train_sets, dev_set

    def get_lm_sets(self):
        params = self.params
        buck_lm_sets = []
        total_lm_files = 0

        for batch_id, batch_size in enumerate(params.lm_buck_batch_size):
            buck_lm_files = glob.glob(path.join(
                params.lm_data_dir, "lm" + str(batch_id) + ".*"))
            total_lm_files += len(buck_lm_files)
            buck_lm_set = LMDataset(buck_lm_files, batch_size)
            buck_lm_sets.append(buck_lm_set)
        print ("Total LM files: %d" %total_lm_files)

        return buck_lm_sets

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
                iter_init_list = []
                iter_handle_list = []
                for train_set in buck_train_sets:
                    iter_init_list.append(train_set.data_iter)
                    iter_handle_list.append(sess.run(train_set.data_iter.string_handle()))


                with tf.variable_scope("model", reuse=None):
                    model = Seq2SeqModel(iterator, True, model_params)
                    # Create eval model

                self.create_eval_model(dev_set)

                if params.lm_prob > 0:
                    # Create LM dataset
                    buck_lm_sets = self.get_lm_sets()
                    lm_handle = tf.placeholder(tf.string, shape=[])
                    lm_iterator = tf.data.Iterator.from_string_handle(
                        lm_handle, buck_lm_sets[0].data_set.output_types,
                        buck_lm_sets[0].data_set.output_shapes)
                    iter_lm_list = []
                    iter_lm_handle_list = []
                    for lm_set in buck_lm_sets:
                        iter_lm_list.append(lm_set.data_iter)
                        iter_lm_handle_list.append(sess.run(lm_set.data_iter.string_handle()))

                    # Create LM model
                    with tf.variable_scope("model", reuse=None):
                        print ("Creating LM model")
                        sys.stdout.flush()
                        lm_params = copy.deepcopy(
                            model_params.decoder_params['char'])
                        lm_params.encoder_hidden_size =\
                            2 * model_params.encoder_params.hidden_size
                        lm_model = LMModel(LMEncoder(lm_params), lm_iterator,
                                           params=params.lm_params)

                model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
                best_model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

                ckpt = tf.train.get_checkpoint_state(params.train_dir)
                if not ckpt:
                    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                    if params.pretrain_lm_path:
                        tf_utils.restore_common_variables(sess, params.pretrain_lm_path)
                    if params.pretrain_phone_path:
                        print ("Oh hello")
                        tf_utils.restore_common_variables(sess, params.pretrain_phone_path)


                else:
                    tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
                # Prepare training data
                epoch = model.global_step.eval()/3000  # The exact number is either 3006 or 3016

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
                if params.lm_prob > 0:
                    lm_steps, lm_loss = 0, 0.0
                previous_errs = []
                try:
                    with open(path.join(params.train_dir, "asr_err.txt"), "r") as err_f:
                        for line in err_f:
                            previous_errs.append(float(line.strip()))
                        print ("Previous perf. log of %d checkpoints loaded" %(len(previous_errs)))
                        if not self.check_progess(previous_errs):
                            print ("No improvement in 10 checkpoints")
                            os._exit(1)
                except:
                    pass

                if params.lm_prob > 0:
                    # Run the LM initializer
                    for iter_lm in iter_lm_list:
                        sess.run(iter_lm.initializer)
                    active_lm_handle_list = copy.deepcopy(iter_lm_handle_list)

                while epoch <= params.max_epochs:
                    print("\nEpochs done: %d" %epoch)
                    sys.stdout.flush()
                    epc_start_time = time.time()

                    active_handle_list = copy.deepcopy(iter_handle_list)
                    for iter_init in iter_init_list:
                        sess.run(iter_init.initializer)

                    while active_handle_list:
                        task = ("lm" if (params.lm_prob > random.random()) else "asr")
                        #task = random.choice(["lm"])
                        if task == "lm":
                            cur_lm_handle = random.choice(active_lm_handle_list)
                            try:
                                output_feed = [lm_model.updates, lm_model.losses]
                                _, lm_step_loss = sess.run(output_feed, feed_dict={lm_handle: cur_lm_handle})
                                lm_loss += lm_step_loss/params.steps_per_checkpoint
                                lm_steps += 1
                                if lm_steps % params.steps_per_checkpoint == 0:
                                    perplexity = math.exp(lm_loss) if lm_loss < 300 else float('inf')
                                    print ("LM steps: %d, Perplexity: %f" %(lm_steps, perplexity))
                                    sys.stdout.flush()

                                    lm_summary = tf_utils.get_summary(perplexity, "LM Perplexity")
                                    train_writer.add_summary(lm_summary, model.global_step.eval())

                                    lm_loss = 0.0
                            except tf.errors.OutOfRangeError:
                                # Run the LM initializer
                                active_lm_handle_list.remove(cur_lm_handle)
                                if not active_lm_handle_list:
                                    print ("LM epoch done!")
                                    for iter_lm in iter_lm_list:
                                        sess.run(iter_lm.initializer)
                                    active_lm_handle_list = copy.deepcopy(iter_lm_handle_list)

                        else:
                            cur_handle = random.choice(active_handle_list)
                            try:
                                output_feed = [model.updates,  model.losses]

                                _, step_loss = sess.run(output_feed, feed_dict={handle: cur_handle})
                                step_loss = step_loss["char"]

                                current_step += 1
                                loss += step_loss / params.steps_per_checkpoint

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
                                    asr_err_cur = self.eval_model.asr_decode(sess)
                                    decode_end_time = time.time() - decode_start_time

                                    print ("ASR error: %.4f, Decoding time: %s"
                                           %(asr_err_cur, timedelta(seconds=decode_end_time)))
                                    sys.stdout.flush()
                                    with open(path.join(params.train_dir, "asr_err.txt"), "a") as err_f:
                                        err_f.write(str(asr_err_cur) + "\n")

                                    err_summary = tf_utils.get_summary(asr_err_cur, "ASR Error")
                                    train_writer.add_summary(err_summary, model.global_step.eval())

                                    previous_errs.append(asr_err_cur)
                                    if model.global_step.eval() >= params.min_steps:
                                        if not self.check_progess(previous_errs, num=6):
                                            # Training has already happened for min epochs and the dev
                                            # error is getting worse w.r.t. the worst value in previous 3 checkpoints
                                            if model.learning_rate.eval() > 1e-4:
                                                sess.run(model.learning_rate_decay_op)
                                                print ("Learning rate decreased !!")
                                                sys.stdout.flush()

                                    if not self.check_progess(previous_errs):
                                        print ("No improvement in 10 checkpoints")
                                        sys.exit()


                                    # Early stopping
                                    if asr_err_best > asr_err_cur:
                                        asr_err_best = asr_err_cur
                                        # Save model
                                        print("Best ASR Error rate: %.4f" % asr_err_best)
                                        print("Saving the best model !!")
                                        sys.stdout.flush()

                                        # Save the best score
                                        f = open(os.path.join(params.train_dir, "best.txt"), "w")
                                        f.write(str(asr_err_best))
                                        f.close()

                                        # Save the model in best model directory
                                        checkpoint_path = os.path.join(params.best_model_dir, "asr.ckpt")
                                        best_model_saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)

                                    # Also save the model for plotting
                                    checkpoint_path = os.path.join(params.train_dir, "asr.ckpt")
                                    model_saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)

                                    print ("\n")
                                    sys.stdout.flush()
                                    # Reinitialze tracking variables
                                    ckpt_start_time = time.time()
                                    loss = 0.0

                            except tf.errors.OutOfRangeError:
                                active_handle_list.remove(cur_handle)


                    print ("Total steps: %d" %model.global_step.eval())
                    sess.run(model.epoch_incr)
                    epoch += 1
                    epc_time = time.time() - epc_start_time
                    print ("\nEPOCH TIME: %s\n" %(str(timedelta(seconds=epc_time))))
                    sys.stdout.flush()


    @classmethod
    def add_parse_options(cls, parser):
        # Training params
        parser.add_argument("-lm_prob", default=0.0, type=float,
                            help="Prob. of running the LM task")
        parser.add_argument("-run_id", "--run_id", default=0, type=int, help="Run ID")
        parser.add_argument("-data_dir", default="/scratch2/asr_multi/data/tfrecords",
                            type=str, help="Data directory")
        parser.add_argument("-lm_data_dir",
                            default="/scratch2/asr_multi/data/tfrecords/lm_all",
                            type=str, help="Data directory")
        parser.add_argument("-vocab_dir", "--vocab_dir", default="/share/data/speech/"
                            "shtoshni/research/datasets/asr_swbd/lang/vocab",
                            type=str, help="Vocab directory")
        parser.add_argument("-tb_dir", "--train_base_dir",
                            default="/scratch2/asr_multi/models",
                            type=str, help="Training directory")
        parser.add_argument("-feat_len", "--feat_length", default=80, type=int,
                            help="Number of features per frame")
        parser.add_argument("-steps_per_checkpoint", default=500,
                            type=int, help="Gradient steps per checkpoint")
        parser.add_argument("-min_steps", "--min_steps", default=25000, type=int,
                            help="Min steps BEFORE DECREASING LEARNING RATE")

        parser.add_argument("-pretrain_lm_path", default="", type=str,
                            help="Pretrain language model path")
        parser.add_argument("-pretrain_phone_path", default="", type=str,
                            help="Pretrain phone model path")

        parser.add_argument("-chaos", default=False, action="store_true",
                            help="Random seed is not controlled if set")


