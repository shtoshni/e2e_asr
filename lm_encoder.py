"""LM Encoder class.

Author: Shubham Toshniwal
Contact: shtoshni@ttic.edu
Date: February, 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
from bunch import Bunch

from base_params import BaseParams
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear


class LMEncoder(BaseParams):
    """Base class for decoder in encoder-decoder framework."""

    @classmethod
    def class_params(cls):
        """Decoder class parameters."""
        params = Bunch()
        params['out_prob'] = 0.9
        params['lm_hidden_size'] = 256
        params['proj_size'] = 256
        params['num_layers'] = 1
        params['emb_size'] = 256
        params['vocab_size'] = 1000

        return params

    def __init__(self, isTraining=True, params=None):
        """The initializer for decoder class.

        Args:
            params: Parameters
        """
        if params is None:
            self.params = self.class_params()
        else:
            self.params = params
        params = self.params
        self.isTraining = isTraining
        self.cell = self.get_cell()

    def get_cell(self):
        """Create the LSTM cell used by decoder."""
        params = self.params
        def single_cell():
            """Create a single RNN cell."""
            cell = tf.nn.rnn_cell.BasicLSTMCell(params.lm_hidden_size)
            if self.isTraining:
                # During training we use a dropout wrapper
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell, output_keep_prob=params.out_prob)
            return cell

        if params.num_layers > 1:
            # If RNN is stacked then we use MultiRNNCell class
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell() for _ in xrange(params.num_layers)])
        else:
            cell = single_cell()

        return cell

    def prepare_decoder_input(self, decoder_inputs):
        """Do this step before starting decoding.

        This step converts the decoder IDs to vectors and
        Args:
            decoder_inputs: Time major decoder IDs
        Returns:
            embedded_inp: Embedded decoder input.
            loop_function: Function for getting next timestep input.
        """
        params = self.params
        with tf.variable_scope("decoder"):
            # Create an embedding matrix
            embedding = tf.get_variable(
                "embedding", [params.vocab_size, params.emb_size],
                initializer=tf.random_uniform_initializer(-1.0, 1.0))
            # Embed the decoder input via embedding lookup operation
            embedded_inp = tf.nn.embedding_lookup(embedding, decoder_inputs)

        return embedded_inp

    def __call__(self, lm_inputs, seq_len):
        """Runs RNN and returns the logits."""
        params = self.params
        emb_inputs = self.prepare_decoder_input(lm_inputs[:-1, :])
        outputs, _ = \
            tf.nn.dynamic_rnn(self.cell, emb_inputs,
                              sequence_length=seq_len,
                              dtype=tf.float32, time_major=True, scope="rnn/lm")
        # T x B x H => (T x B) x H
        outputs = tf.reshape(outputs, [-1, self.cell.output_size])

        with tf.variable_scope("rnn"):
            # Additional variable scope required to mimic the attention
            # decoder scope so that variable initialization is hassle free
            if params.lm_hidden_size != params.proj_size:
                with tf.variable_scope("SimpleProjection"):
                    outputs = _linear([outputs], params.proj_size, True)

            with tf.variable_scope("OutputProjection"):
                outputs = _linear([outputs], params.vocab_size, True)

        return outputs
