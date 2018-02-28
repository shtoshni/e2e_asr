# coding: utf-8

from __future__ import absolute_import
from __future__ import division

import math
import os

from os import path
import time

import argparse

import numpy as np
from bunch import Bunch
import editdistance as ed
import tensorflow as tf

import data_utils
from base_params import BaseParams
import swbd_utils


class Eval(BaseParams):

    @classmethod
    def class_params(cls):
        params = Bunch()
        params['best_model_dir'] = "/scratch"
        params['vocab_dir'] = "/share/data/speech/shtoshni/research/datasets/asr_swbd/lang/vocab"
        return params


    def __init__(self, model, params=None):
        if params is None:
            self.params = params
        else:
            self.params = params

        self.model = model
        self.rev_phone_vocab = self.load_phone_vocab()

    def load_phone_vocab(self):
        phone_vocab_path = path.join(self.params.vocab_dir, "phone.vocab")
        _, rev_phone_vocab = data_utils.initialize_vocabulary(phone_vocab_path)
        return rev_phone_vocab

    def phone_decode(self, sess):
        params = self.params
        # Load vocabularies.

        rev_normalizer = swbd_utils.reverse_swbd_normalizer()

        gold_asr_file = path.join(params.best_model_dir, 'gold_phones.txt')
        decoded_asr_file = path.join(params.best_model_dir, 'decoded_phones.txt')

        total_errors, total_phones = 0, 0
        # Initialize the dev iterator
        sess.run(self.model.data_iter.initializer)

        with open(gold_asr_file, 'w') as gold_f, \
                open(decoded_asr_file, 'w') as proc_dec_f:
            while True:
                try:
                    output_feed = [self.model.decoder_inputs["utt_id"],
                                   self.model.decoder_inputs["phone"],
                                   self.model.outputs["phone"]]

                    utt_ids, gold_ids, output_logits = sess.run(output_feed)

                    gold_ids = np.array(gold_ids[1:, :]).T
                    batch_size = gold_ids.shape[0]

                    outputs = np.argmax(output_logits, axis=1)
                    outputs = np.reshape(outputs, (-1, batch_size))  # T*B

                    to_decode = outputs.T  # B*T

                    for sent_id in xrange(batch_size):
                        gold_phones = self.phone_array_to_sent(
                            gold_ids[sent_id, :], self.rev_phone_vocab)
                        decoded_phones = self.phone_array_to_sent(
                            to_decode[sent_id, :], self.rev_phone_vocab)

                        total_errors += ed.eval(gold_phones, decoded_phones)
                        total_phones += len(gold_phones)

                        gold_f.write(utt_ids[sent_id] + '\t' +
                                     '{}\n'.format(' '.join(gold_phones)))
                        proc_dec_f.write(utt_ids[sent_id] + '\t' +
                                         '{}\n'.format(' '.join(decoded_phones)))

                except tf.errors.OutOfRangeError:
                    break
        try:
            score = float(total_errors)/float(total_phones)
        except ZeroDivisionError:
            score = 0.0

        print ("Output at: %s" %str(decoded_asr_file))
        return score

    @staticmethod
    def phone_array_to_sent(phone_array, reverse_phone_vocab):
        phone_id_list = list(phone_array)
        if data_utils.EOS_ID in phone_id_list:
            phone_id_list = phone_id_list[:phone_id_list.index(data_utils.EOS_ID)]
        phone_list = map(lambda phone_id:
                      tf.compat.as_str(reverse_phone_vocab[phone_id]), phone_id_list)
        phone_sent = (' '.join(phone_list)).strip()
        return phone_sent
