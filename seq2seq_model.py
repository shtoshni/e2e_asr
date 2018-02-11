"""Seq2Seq model class that creates the computation graph.

Author: Shubham Toshniwal
Date: February, 2018
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
from bunch import Bunch

import tensorflow as tf

import data_utils
from losses import LossUtils


class Seq2SeqModel(object):
    """Implements the Attention-Enabled Encoder-Decoder model."""

    @classmethod
    def class_params(cls):
        params = Bunch()
        params['batch_size'] = 64
        params['isTraining'] = True
        # Task specification
        params['buckets'] = {'char':[(210, 60), (346, 120), (548, 180), (850, 200),
                                     (1500, 380)],
                             'phone': [(210, 50), (346, 110), (548, 140), (850, 150),
                                       (1500, 250)]}
        params['tasks'] = ['char']
        params['num_layers'] = {'char':1}
        params['feat_length'] = 80

        # Optimization params
        params['learning_rate'] = 1e-3
        params['learning_rate_decay_factor'] = 0.9
        params['max_gradient_norm'] = 5.0

        # Loss params
        params['avg'] = True

        return params

    def __init__(self, encoder, decoder, data_iter, params=None):
        """Initializer of class that defines the computational graph.

        Args:
            encoder: Encoder object executed via encoder(args)
            decoder: Decoder object executed via decoder(args)
        """
        if params is None:
            self.params = self.class_params()
        else:
            self.params = params

        self.data_iter = data_iter

        self.learning_rate = tf.Variable(float(params.learning_rate),
                                         trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * params.learning_rate_decay_factor)

        # Number of gradient updates performed
        self.global_step = tf.Variable(0, trainable=False)
        # Number of epochs done
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_incr = self.epoch.assign(self.epoch + 1)

        self.encoder = encoder
        self.decoder = decoder

        self.create_computational_graph()

        # Model saver function
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

    def create_computational_graph(self):
        """Creates the computational graph."""
        params = self.params
        self.encoder_inputs, self.decoder_inputs, self.seq_len, \
            self.seq_len_target = self.get_batch(self.data_iter.get_next())

        self.targets = {}
        self.target_weights = {}
        for task in params.tasks:
            # Targets are shifted by one - T*B
            self.targets[task] = tf.slice(self.decoder_inputs[task], [1, 0], [-1, -1])

            batch_major_mask = tf.sequence_mask(self.seq_len_target[task],
                                                dtype=tf.float32)  # B*T
            time_major_mask = tf.transpose(batch_major_mask, [1, 0])  # T*B
            self.target_weights[task] = tf.reshape(time_major_mask, [-1])

        # Create computational graph
        # First encode input
        self.encoder_hidden_states, self.time_major_states, self.seq_len_encs =\
            self.encoder(self.encoder_inputs, self.seq_len, params.num_layers)

        self.outputs = {}
        for task in params.tasks:
            task_depth = params.num_layers[task]
            # Then decode
            self.outputs[task] = self.decoder[task](
                self.decoder_inputs[task], self.seq_len_target[task],
                self.encoder_hidden_states[task_depth], self.seq_len_encs[task_depth])

        if params.isTraining:
            self.losses = {}
            for task in params.tasks:
                task_depth = params.num_layers[task]
                # Training outputs and losses.
                self.losses[task] = LossUtils.seq2seq_loss(
                    self.outputs[task], self.targets[task], self.seq_len_target[task])

            tf.summary.scalar('Negative log likelihood ' + task, self.losses[task])
            # Gradients and parameter updation for training the model.
            trainable_vars = tf.trainable_variables()
            print ("\nModel parameters:\n")
            for var in trainable_vars:
                print (("{0}: {1}").format(var.name, var.get_shape()))
            print ("\n")
            # Initialize optimizer
            opt = tf.train.AdamOptimizer(self.learning_rate)

            # Add losses across the tasks
            self.total_loss = 0.0
            for task in params.tasks:
                self.total_loss += self.losses[task]

            if params.avg:
                self.total_loss /= float(len(params.tasks))
            tf.summary.scalar('Total loss', self.total_loss)
            # Get gradients from loss
            gradients = tf.gradients(self.total_loss, trainable_vars)
            # Gradient clipping
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             params.max_gradient_norm)
            self.gradient_norm = norm
            tf.summary.scalar('Gradient Norm', self.gradient_norm)
            # Apply gradients
            self.updates = opt.apply_gradients(
                zip(clipped_gradients, trainable_vars),
                global_step=self.global_step)


    def get_batch(self, batch):
        encoder_inputs = batch["logmel"]
        encoder_len = batch["logmel_len"]

        decoder_inputs = {}
        decoder_len = {}
        for task in self.params.num_layers:
            decoder_inputs[task] = tf.transpose(batch[task], [1, 0])
            decoder_len[task] = batch[task + "_len"]
            if not self.params.isTraining:
                decoder_len[task] = tf.ones_like(decoder_len[task]) * 100

        return [encoder_inputs, decoder_inputs, encoder_len, decoder_len]


    @staticmethod
    def get_instance(proto, feat_length=80):
        # -- get reader and read serialized examples from queue
        context_features = {
            "segment": tf.FixedLenFeature([], tf.string),
            "logmel_len": tf.FixedLenFeature([], tf.int64),
            "cint_len": tf.FixedLenFeature([], tf.int64),
            "pint_len": tf.FixedLenFeature([], tf.int64),
        }
        sequence_features = {
            "logmel": tf.FixedLenSequenceFeature(shape=[feat_length], dtype=tf.float32),
            "cint": tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64),
            "pint": tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64)
        }
        # ---------------------------------------------------------------------------
        # -- parse a sequence example given the above instructions on the structure
        context,sequence = tf.parse_single_sequence_example(
            serialized=proto,
            context_features=context_features,
            sequence_features=sequence_features
        )
        # ---------------------------------------------------------------------------
        # -- unpack segment ID
        segmentID = context["segment"]
        # -- ready batch of speech // characters // phonemes
        logmel = sequence["logmel"]
        cint = sequence["cint"]
        pint = sequence["pint"]
        # -- get (non-zero) lengths of sequences
        logmel_len = context["logmel_len"]
        cint_len = context["cint_len"]
        pint_len = context["pint_len"]

        return {"logmel": logmel, "char": cint, "phone":pint,
                "logmel_len": logmel_len, "char_len": cint_len,
                "phone_len": pint_len}
