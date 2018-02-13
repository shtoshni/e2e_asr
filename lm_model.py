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


class LMModel(object):
    """Implements the Attention-Enabled Encoder-Decoder model."""

    @classmethod
    def class_params(cls):
        params = Bunch()

        # Optimization params
        params['learning_rate'] = 1e-3
        params['learning_rate_decay_factor'] = 0.5
        params['max_gradient_norm'] = 5.0

        return params

    def __init__(self, encoder, data_iter, params=None):
        """Initializer of class that defines the computational graph.

        Args:
            encoder: Encoder object executed via encoder(args)
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

        self.create_computational_graph()

    def create_computational_graph(self):
        """Creates the computational graph."""
        params = self.params
        self.encoder_inputs, self.seq_len = self.data_iter.get_batch()

        self.targets, self.target_weights =\
            tf_utils.create_shifted_targets(self.encoder_inputs, self.seq_len)

        # Create computational graph
        # First encode input
        self.outputs = self.encoder(self.encoder_inputs, self.seq_len)
        self.loss = LossUtils.cross_entropy_loss(
            self.outputs, self.targets, self.seq_len)

        # Gradients and parameter updation for training the model.
        trainable_vars = tf.trainable_variables()
        total_params = 0
        print ("\nShared parameters:\n")
        for var in trainable_vars:
            print (("{0}: {1}").format(var.name, var.get_shape()))
            var_params = 1
            for dim in var.get_shape().as_list():
                var_params *= dim
            total_params += var_params

        # Initialize optimizer
        opt = tf.train.AdamOptimizer(self.learning_rate)

        # Get gradients from loss
        gradients = tf.gradients(self.loss, trainable_vars)
        # Gradient clipping
        clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                         params.max_gradient_norm)
        # Apply gradients
        self.updates = opt.apply_gradients(
            zip(clipped_gradients, trainable_vars),
            global_step=self.global_step)
