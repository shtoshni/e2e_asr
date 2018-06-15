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

import tf_utils
import data_utils
from losses import LossUtils
from base_params import BaseParams

from encoder import Encoder
from attn_decoder import AttnDecoder


class Seq2SeqModel(BaseParams):
    """Implements the Attention-Enabled Encoder-Decoder model."""

    @classmethod
    def class_params(cls):
        params = Bunch()
        # Task specification
        params['tasks'] = ['char']
        params['num_layers'] = {'char': 4}
        params['max_output'] = {'char': 120}
        params['label_smoothing'] = 0.05

        # Optimization params
        params['learning_rate'] = 1e-3
        params['learning_rate_decay_factor'] = 0.5
        params['max_gradient_norm'] = 5.0

        # Loss params
        params['avg'] = True

        params['encoder_params'] = Encoder.class_params()
        params['decoder_params'] = {'char': AttnDecoder.class_params()}

        return params

    def __init__(self, data_iter, isTraining=True, params=None):
        """Initializer of class that defines the computational graph.

        Args:
            encoder: Encoder object executed via encoder(args)
            decoder: Decoder object executed via decoder(args)
        """
        if params is None:
            self.params = self.class_params()
        else:
            self.params = params
        params = self.params

        self.encoder = Encoder(isTraining=isTraining,
                               params=params.encoder_params)
        self.decoder = {}
        for task in params.tasks:
            self.decoder[task] = AttnDecoder(isTraining=isTraining,
                                             params=params.decoder_params[task],
                                             scope=task)
        self.data_iter = data_iter

        self.isTraining = isTraining

        self.learning_rate = tf.Variable(float(params.learning_rate),
                                         trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * params.learning_rate_decay_factor)

        # Number of gradient updates performed
        self.global_step = tf.Variable(0, trainable=False)
        # Number of epochs done
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_incr = self.epoch.assign(self.epoch + 1)


        self.create_computational_graph()

    def create_computational_graph(self):
        """Creates the computational graph."""
        params = self.params
        self.encoder_inputs, self.decoder_inputs, self.seq_len, \
            self.seq_len_target = self.get_batch(self.data_iter.get_next())

        self.targets = {}
        self.target_weights = {}
        for task in params.tasks:
            # Targets are shifted by one - T*B
            self.targets[task], self.target_weights[task] =\
                tf_utils.create_shifted_targets(self.decoder_inputs[task],
                                                self.seq_len_target[task])

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

        if self.isTraining:
            self.losses = {}
            for task in params.tasks:
                task_depth = params.num_layers[task]
                # Training outputs and losses.
                self.losses[task] = LossUtils.smooth_cross_entropy_loss(
                    self.outputs[task], self.targets[task], self.decoder[task].params.vocab_size,
                    self.seq_len_target[task], label_smoothing=params.label_smoothing)

            tf.summary.scalar('Negative log likelihood ' + task, self.losses[task])
            # Gradients and parameter updation for training the model.
            trainable_vars = tf.trainable_variables()
            # Remove the LM LSTM params
            trainable_vars = [var for var in trainable_vars if ("/lm/" not in var.name)
                              and ("/SimpleProjection/" not in var.name) and
                              ("/OutputProjection/" not in var.name) and
                              ("/embedding" not in var.name)]

            total_params = 0
            print ("\nModel parameters:\n")
            for var in trainable_vars:
                print (("{0}: {1}").format(var.name, var.get_shape()))
                var_params = 1
                for dim in var.get_shape().as_list():
                    var_params *= dim
                total_params += var_params

            print ("\nFrozen params:\n")
            lm_vars = list(set(tf.trainable_variables()).difference(set(trainable_vars)))
            for var in lm_vars:
                print (("{0}: {1}").format(var.name, var.get_shape()))
                var_params = 1
                for dim in var.get_shape().as_list():
                    var_params *= dim
                total_params += var_params

            print ("\nTOTAL PARAMS: %.2f (in millions)\n" %(total_params/1e6))

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
            # Apply gradients
            self.updates = opt.apply_gradients(
                zip(clipped_gradients, trainable_vars),
                global_step=self.global_step)
            # Summary merger
            self.merged = tf.summary.merge_all()

    def get_batch(self, batch):
        """Get a batch from the iterator."""
        encoder_inputs = batch["logmel"]
        encoder_len = batch["logmel_len"]

        if self.encoder.params.stack_cons > 1:
            feat_size = encoder_inputs.get_shape()[2].value
            # Remove delta coeffs
            #feat_size_no_del = feat_size // 2

            #stacking_tens = [encoder_inputs[:, :, feat_size_no_del:]]
            #batch_size = tf.shape(encoder_inputs)[0]
            #for shift in xrange(1, self.encoder.params.stack_cons):
            #    shifted_inp = tf.concat([encoder_inputs[:, shift:, feat_size_no_del:],
            #                            tf.zeros([batch_size, shift, feat_size_no_del])],  1)
            #    stacking_tens.append(shifted_inp)

            stacking_tens = [encoder_inputs]
            batch_size = tf.shape(encoder_inputs)[0]
            for shift in xrange(1, self.encoder.params.stack_cons):
                shifted_inp = tf.concat([encoder_inputs[:, shift:, :],
                                         tf.zeros([batch_size, shift, feat_size])],  1)
                stacking_tens.append(shifted_inp)

            encoder_inputs = tf.concat(stacking_tens, 2)


        decoder_inputs = {}
        decoder_len = {}
        for task in self.params.tasks:
            decoder_inputs[task] = tf.transpose(batch[task], [1, 0])
            decoder_len[task] = batch[task + "_len"]
            if not self.isTraining:
                decoder_len[task] = tf.ones_like(decoder_len[task]) *\
                    self.params.max_output[task]

        if not self.isTraining:
            decoder_inputs["utt_id"] = batch["utt_id"]
        return [encoder_inputs, decoder_inputs, encoder_len, decoder_len]

    @classmethod
    def add_parse_options(cls, parser):
        # Seq2Seq params
        parser.add_argument("-tasks", "--tasks", default="", type=str,
                            help="Auxiliary task choices")
        parser.add_argument("-nlc", "--num_layers_char", default=4, type=int,
                            help="Output layer of encoder which is used for char.")
        parser.add_argument("-nlp", "--num_layers_phone", default=3, type=int,
                            help="Output layer of encoder which is used for phone.")
        parser.add_argument("-max_out_char", "--max_output_char", default=120,
                            type=int, help="Maximum length of char/word-piece sequence")
        parser.add_argument("-max_out_phone", "--max_output_phone", default=250,
                            type=int, help="Maximum length of phone sequence")
        # Regularization params
        parser.add_argument("-label_smoothing", default=0.1,
                            type=float, help="Label smoothing")
        # Optimization params
        parser.add_argument("-lr_decay", "--learning_rate_decay_factor", default=0.5,
                            type=float, help="Learning rate decay factor")
        parser.add_argument("-avg", "--avg", default=False, action="store_true",
                            help="Average the loss")
