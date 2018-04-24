"""Language model class that creates the computation graph.

Author: Shubham Toshniwal
Date: February, 2018
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from bunch import Bunch

import random
import tensorflow as tf

import tf_utils
from losses import LossUtils
from base_params import BaseParams
from lm_dataset import LMDataset
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear


class LMModel(BaseParams):
    """Language model."""

    @classmethod
    def class_params(cls):
        params = Bunch()

        # Optimization params
        params['lm_batch_size'] = 128
        params['lm_learning_rate'] = 1e-4
        params['lm_learning_rate_decay_factor'] = 0.5
        params['max_gradient_norm'] = 5.0
        params['simple_lm'] = False

        return params

    def __init__(self, encoder, data_files, params=None):
        """Initializer of class

        Args:
            encoder: Encoder object executed via encoder(args)
        """
        if params is None:
            self.params = self.class_params()
        else:
            self.params = params
        params = self.params

        self.data_files = data_files
        self.data_iter = self.update_iterator()

        self.learning_rate = tf.Variable(float(params.lm_learning_rate),
                                         trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * params.lm_learning_rate_decay_factor)

        # Number of gradient updates performed
        self.lm_global_step = tf.Variable(0, trainable=False)
        # Number of epochs done
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_incr = self.epoch.assign(self.epoch + 1)

        self.encoder = encoder

        self.create_computational_graph()
        # Gradients and parameter updation for training the model.
        trainable_vars = tf.trainable_variables()#[]
        #for var in tf.trainable_variables():
        #    if "decoder_char" in var.name:
        #        trainable_vars.append(var)
        #        print (var.name)

        # Initialize optimizer
        opt = tf.train.AdamOptimizer(self.learning_rate, name='AdamLM')

        # Get gradients from loss
        gradients = tf.gradients(self.losses, trainable_vars)
        # Gradient clipping
        clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                      params.max_gradient_norm)
        # Apply gradients
        self.updates = opt.apply_gradients(
            zip(clipped_gradients, trainable_vars),
            global_step=self.lm_global_step)

    def update_iterator(self):
        """Create data iterator."""
        random.shuffle(self.data_files)
        lm_set = LMDataset(self.data_files, self.params.lm_batch_size)
        return lm_set.data_iter

    def create_computational_graph(self):
        """Creates the computational graph."""
        self.encoder_inputs, self.seq_len = self.get_batch()

        self.targets, self.target_weights =\
            tf_utils.create_shifted_targets(self.encoder_inputs, self.seq_len)
        # Create computational graph
        # First encode input
        with tf.variable_scope("rnn_decoder_char", reuse=tf.AUTO_REUSE):
            self.outputs = self.encoder(self.encoder_inputs, self.seq_len)

        self.losses = LossUtils.cross_entropy_loss(
            self.outputs, self.targets, self.seq_len)

    def get_batch(self):
        """Get a batch from the iterator."""
        batch = self.data_iter.get_next()
        # (T + 1) * B - encoder_len would take care of not processing (T+1)th symbol
        encoder_inputs = tf.transpose(batch["char"], [1, 0])
        encoder_len = batch["char_len"]

        return [encoder_inputs, encoder_len]

    @classmethod
    def add_parse_options(cls, parser):
        # LM params
        parser.add_argument("-lm_learning_rate", default=0.0001, type=float,
                            help="LM learning rate")
