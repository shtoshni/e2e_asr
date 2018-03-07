# coding: utf-8

from __future__ import absolute_import
from __future__ import division

import math
import os

from os import path
import time

import argparse

import numpy as np
from bunch import Bunch, bunchify
import editdistance as ed
import tensorflow as tf

import data_utils
from seq2seq_model import Seq2SeqModel
from speech_dataset import SpeechDataset
from base_params import BaseParams
import swbd_utils

from beam_search import BeamSearch

class Eval(BaseParams):

    @classmethod
    def class_params(cls):
        params = Bunch()
        params['beam_size'] = 1
        params['best_model_dir'] = "/scratch"
        params['vocab_dir'] = "/share/data/speech/shtoshni/research/datasets/asr_swbd/lang/vocab"
        return params


    def __init__(self, model, params=None):
        if params is None:
            self.params = params
        else:
            self.params = params

        self.model = model
        self.rev_char_vocab = self.load_char_vocab()

    def load_char_vocab(self):
        char_vocab_path = path.join(self.params.vocab_dir, "char.vocab")
        _, rev_char_vocab = data_utils.initialize_vocabulary(char_vocab_path)
        return rev_char_vocab

    def asr_decode(self, sess):
        params = self.params

        rev_normalizer = swbd_utils.reverse_swbd_normalizer()

        gold_asr_file = path.join(params.best_model_dir, 'gold_asr.txt')
        decoded_asr_file = path.join(params.best_model_dir, 'decoded_asr.txt')
        raw_asr_file = path.join(params.best_model_dir, 'raw_asr.txt')

        total_errors, total_words = 0, 0
        # Initialize the dev iterator
        sess.run(self.model.data_iter.initializer)

        with open(gold_asr_file, 'w') as gold_f, open(raw_asr_file, 'w') as raw_dec_f,\
                open(decoded_asr_file, 'w') as proc_dec_f:
            while True:
                try:
                    output_feed = [self.model.decoder_inputs["utt_id"],
                                   self.model.decoder_inputs["char"],
                                   self.model.outputs["char"]]

                    utt_ids, gold_ids, output_logits \
                        = sess.run(output_feed)

                    gold_ids = np.array(gold_ids[1:, :]).T
                    batch_size = gold_ids.shape[0]

                    outputs = np.argmax(output_logits, axis=1)
                    outputs = np.reshape(outputs, (-1, batch_size))  # T*B

                    to_decode = outputs.T  # B*T

                    for sent_id in xrange(batch_size):
                        gold_asr = self.wp_array_to_sent(
                            gold_ids[sent_id, :], self.rev_char_vocab, rev_normalizer)
                        decoded_asr = self.wp_array_to_sent(
                            to_decode[sent_id, :], self.rev_char_vocab, rev_normalizer)
                        raw_asr_words, decoded_words = data_utils.get_relevant_words(decoded_asr)
                        _, gold_words = data_utils.get_relevant_words(gold_asr)

                        total_errors += ed.eval(gold_words, decoded_words)
                        total_words += len(gold_words)

                        gold_f.write(utt_ids[sent_id] + '\t' +
                                     '{}\n'.format(' '.join(gold_words)))
                        raw_dec_f.write(utt_ids[sent_id] + '\t' +
                                        '{}\n'.format(' '.join(raw_asr_words)))
                        proc_dec_f.write(utt_ids[sent_id] + '\t' +
                                         '{}\n'.format(' '.join(decoded_words)))

                except tf.errors.OutOfRangeError:
                    break
        try:
            score = float(total_errors)/float(total_words)
        except ZeroDivisionError:
            score = 0.0

        print ("Output at: %s" %str(raw_asr_file))
        print ("Score: %f" %score)
        return score

    def beam_search_decode(self, sess, ckpt_path):
        """Beam search decoding done via numpy implementation of attention decoder."""
        params = self.params

        beam_search = BeamSearch(ckpt_path, self.rev_char_vocab, beam_size=params.beam_size)
        rev_normalizer = swbd_utils.reverse_swbd_normalizer()

        gold_asr_file = path.join(params.best_model_dir, 'gold.txt')
        decoded_asr_file = path.join(params.best_model_dir, 'decoded_' +
                                     str(params.beam_size) + '.txt')
        raw_asr_file = path.join(params.best_model_dir, 'raw_' +
                                 str(params.beam_size) + '.txt')

        total_errors, total_words = 0, 0
        # Initialize the dev iterator
        sess.run(self.model.data_iter.initializer)

        counter = 0
        with open(gold_asr_file, 'w') as gold_f, open(raw_asr_file, 'w') as raw_dec_f,\
                open(decoded_asr_file, 'w') as proc_dec_f:
            while True:
                try:
                    output_feed = [self.model.encoder_hidden_states[self.model.params.num_layers["char"]],
                                   self.model.decoder_inputs["utt_id"],
                                   self.model.decoder_inputs["char"]]

                    encoder_hidden_states, utt_ids, gold_ids = sess.run(output_feed)

                    gold_ids = np.array(gold_ids[1:, 0])  # Ignore the GO_ID

                    beam_output = beam_search(encoder_hidden_states[0])
                    gold_asr = self.wp_array_to_sent(
                        gold_ids, self.rev_char_vocab, rev_normalizer)
                    decoded_asr = self.wp_array_to_sent(
                        beam_output, self.rev_char_vocab, rev_normalizer)
                    raw_asr_words, decoded_words = data_utils.get_relevant_words(decoded_asr)
                    _, gold_words = data_utils.get_relevant_words(gold_asr)

                    total_errors += ed.eval(gold_words, decoded_words)
                    total_words += len(gold_words)

                    gold_f.write(utt_ids[0] + '\t' +
                                 '{}\n'.format(' '.join(gold_words)))
                    raw_dec_f.write(utt_ids[0] + '\t' +
                                    '{}\n'.format(' '.join(raw_asr_words)))
                    proc_dec_f.write(utt_ids[0] + '\t' +
                                     '{}\n'.format(' '.join(decoded_words)))
                    counter += 1

                    if counter % 100 == 0:
                        print ("Counter: %d" %counter)
                        #break

                except tf.errors.OutOfRangeError:
                    break
        try:
            score = float(total_errors)/float(total_words)
        except ZeroDivisionError:
            score = 0.0

        print ("Output at: %s" %str(raw_asr_file))
        print ("Score: %f" %score)
        return score

    @staticmethod
    def wp_array_to_sent(wp_array, reverse_char_vocab, normalizer):
        """Convert word piece ID list to sentence."""
        wp_id_list = list(wp_array)
        if data_utils.EOS_ID in wp_id_list:
            wp_id_list = wp_id_list[:wp_id_list.index(data_utils.EOS_ID)]
        wp_list = [tf.compat.as_str(reverse_char_vocab[piece_id])
                   for piece_id in wp_id_list]
        sent = (''.join(wp_list).replace('▁', ' ')).strip()
        return normalizer(sent)
