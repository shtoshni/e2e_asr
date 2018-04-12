# coding: utf-8

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

import numpy as np
from bunch import Bunch, bunchify
import editdistance as ed
import tensorflow as tf

import data_utils
from attn_decoder import AttnDecoder
from encoder import Encoder
from eval_model import Eval
from lm_encoder import LMEncoder
from lm_model import LMModel
from seq2seq_model import Seq2SeqModel
from speech_dataset import SpeechDataset
from lm_dataset import LMDataset
from train import Train
from beam_search import BeamSearch

def parse_options():
    parser = argparse.ArgumentParser()

    Train.add_parse_options(parser)
    Encoder.add_parse_options(parser)
    AttnDecoder.add_parse_options(parser)
    Seq2SeqModel.add_parse_options(parser)
    LMModel.add_parse_options(parser)

    BeamSearch.add_parse_options(parser)

    parser.add_argument("-eval_dev", default=False, action="store_true",
                        help="Get dev set results using the last saved model")
    parser.add_argument("-test", default=False, action="store_true",
                        help="Get test results using the last saved model")
    args = parser.parse_args()
    args = vars(args)
    return process_args(args)


def process_args(options):
    """Process arguments."""

    def get_train_dir(options):
        """Get train directory name given the options."""
        num_layer_string = ""
        for task in options['tasks']:
            if task == "char":
                continue
            num_layer_string += task + "_" + str(options['num_layers_' + task]) + "_"

        skip_string = ""
        if options['skip_step'] != 1:
            skip_string = "skip_" + str(options['skip_step']) + "_"

        train_dir = (skip_string +
                     num_layer_string +
                     ('lstm_' if options['use_lstm'] else '') +
                     (('stack_' + str(options['stack_cons']) + "_")
                      if options['stack_cons'] > 1 else '') +
                     (('base_stride_' + str(options['initial_res_fac'])  + "_")
                      if options['initial_res_fac'] > 1 else '') +
                     (('char_dec_dep_' + str(options['num_layers_dec']) + '_')
                      if options['num_layers_dec'] > 1 else '') +
                     ('lm_prob_' + str(options['lm_prob']) + '_') +
                     'run_id_' + str(options['run_id']) +
                     ('_avg_' if options['avg'] else '')
        )
        return train_dir

    def parse_tasks(task_string):
        tasks = ["char"]
        if "p" in task_string:
            tasks.append("phone")
        return tasks

    options['tasks'] = parse_tasks(options['tasks'])

    train_dir = get_train_dir(options)
    options['train_dir'] = os.path.join(options['train_base_dir'], train_dir)
    options['best_model_dir'] = os.path.join(
        os.path.join(options['train_base_dir'], "best_models"), train_dir)

    for key_prefix in ['num_layers', 'max_output']:
        comb_dict = {}
        for task in options['tasks']:
            comb_dict[task] = options[key_prefix + "_" + task]
        options[key_prefix] = comb_dict

    options['vocab_size'] = {}
    for task in options['tasks']:
        target_vocab, _ = data_utils.initialize_vocabulary(
            os.path.join(options['vocab_dir'], task + ".vocab"))

        options['vocab_size'][task] = len(target_vocab)

    # Process training/eval params
    train_params = Train.get_updated_params(options)
    # Process beam search params
    beam_search_params = BeamSearch.get_updated_params(options)
    # Process model params
    encoder_params = Encoder.get_updated_params(options)
    decoder_params_base = AttnDecoder.get_updated_params(options)
    decoder_params = {}
    for task in options['tasks']:
        task_params = copy.deepcopy(decoder_params_base)
        task_params.vocab_size = options['vocab_size'][task]
        task_params.max_output = options['max_output'][task]
        if task is not "char":
            # Only make the char model deep
            task_params.num_layers_dec = 1

        decoder_params[task] = task_params

    seq2seq_params = Seq2SeqModel.get_updated_params(options)
    seq2seq_params.encoder_params = encoder_params
    seq2seq_params.decoder_params = decoder_params

    lm_params = LMModel.get_updated_params(options)
    train_params.lm_params = lm_params

    if not options['test'] and not options['eval_dev']:
        if not os.path.exists(options['train_dir']):
            os.makedirs(options['train_dir'])
            os.makedirs(options['best_model_dir'])

        # Sort the options to create a parameter file
        parameter_file = 'parameters.txt'
        sorted_args = sorted(options.items(), key=operator.itemgetter(0))

        with open(os.path.join(options['train_dir'], parameter_file), 'w') as g:
            for arg, arg_val in sorted_args:
                sys.stdout.write(arg + "\t" + str(arg_val) + "\n")
                sys.stdout.flush()
                g.write(arg + "\t" + str(arg_val) + "\n")

    proc_options = Bunch()
    proc_options.train_params = train_params
    proc_options.beam_search_params = beam_search_params
    proc_options.seq2seq_params = seq2seq_params
    proc_options.eval_dev = options['eval_dev']
    proc_options.test = options['test']

    return proc_options


def launch_train(options):
    """Launches training of model."""
    trainer = Train(options.seq2seq_params, options.train_params)
    trainer.train()


def launch_eval(options):
    with tf.Session() as sess:
        trainer = Train(options.seq2seq_params, options.train_params)
        if options.eval_dev:
            _, dev_set = trainer.get_data_sets()
        else:
            dataset_params = Bunch()
            dataset_params.batch_size = 64
            dataset_params.feat_length = options.train_params.feat_length

            #test_files = glob.glob(path.join(options.train_params.data_dir, "eval2000*"))
            test_files = glob.glob(path.join(options.train_params.data_dir, "dev_1k.3*"))
            print ("Total test files: %d" %len(test_files))
            dev_set = SpeechDataset(dataset_params, test_files,
                                    isTraining=False)

        with tf.variable_scope("model"):
            print ("Creating dev model")
            dev_seq2seq_params = copy.deepcopy(options.seq2seq_params)
            dev_seq2seq_params.tasks = {'char'}
            dev_seq2seq_params.num_layers = {'char': dev_seq2seq_params.num_layers['char']}
            model_dev = Seq2SeqModel(dev_set.data_iter, isTraining=False,
                                     params=dev_seq2seq_params)

            params = Bunch()
            params.best_model_dir = trainer.params.best_model_dir
            params.vocab_dir = trainer.params.vocab_dir

            eval_model = Eval(model_dev, params=params)

        ckpt = tf.train.get_checkpoint_state(options.train_params.train_dir)
        ckpt_best = tf.train.get_checkpoint_state(options.train_params.best_model_dir)
        ckpt_path = None
        if ckpt_best:
            ckpt_path = ckpt_best.model_checkpoint_path
            tf.train.Saver().restore(sess, ckpt_path)
        elif ckpt:
            ckpt_path = ckpt.model_checkpoint_path
            tf.train.Saver().restore(sess, ckpt_path)
        else:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        print ("Using the model from: %s" %ckpt_path)
        start_time = time.time()
        asr_perf, out_file = eval_model.beam_search_decode(
            sess, ckpt_path, beam_search_params=options.beam_search_params, dev=options.eval_dev,
            get_out_file=True)
        decoding_time = time.time() - start_time
        print ("Total decoding time: %s" %timedelta(seconds=decoding_time))

        return asr_perf, out_file


if __name__ == "__main__":
    options = parse_options()
    if options.eval_dev or options.test:
        launch_eval(options)
    else:
        launch_train(options)
