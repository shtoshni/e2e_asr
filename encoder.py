"""Encoder class of the seq2seq model.

Author: Shubham Toshniwal
Contact: shtoshni@ttic.edu
Date: February, 2018
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
from six.moves import zip
from bunch import Bunch
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
import tensorflow.contrib.rnn as rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear
from tensorflow.python.ops.rnn_cell_impl import _linear as linear
import tensorflow.contrib.cudnn_rnn as cudnn_rnn
import tensorflow as tf


class Encoder(object):
    """Encoder class that encodes input sequence."""

    @classmethod
    def class_params(cls):
        """Decoder class parameters."""
        params = Bunch()
        params['isTraining'] = True
        params['out_prob'] = 0.7
        params['hidden_size'] = 256
        params['bi_dir'] = True
        params['skip_step'] = 2  # Pyramidal architecture
        params['initial_res_fac'] = 1

        return params

    def __init__(self, params=None):
        """Initializer for encoder class."""
        if params is not None:
            self.params = params
        else:
            self.params = self.class_params()
        params = self.params

        self.cell = rnn_cell.BasicLSTMCell(params.hidden_size)
        if params.isTraining:
            # During training a dropout wrapper is used
            self.cell = rnn_cell.DropoutWrapper(
                self.cell, output_keep_prob=params.out_prob)


    def _layer_encoder_input(self, encoder_inputs, seq_len, layer_depth=1):
        """Run a single LSTM on given input.

        Args:
            encoder_inputs: A 3-D Tensor input of shape TxBxE on which to run
                LSTM where T is number of timesteps, B is batch size and E is
                input dimension at each timestep.
            seq_len: A 1-D tensor that contains the actual length of each
                input in the batch. This ensures pad symbols are not
                processed as input.
            layer_depth: A integer denoting the depth at which the current
                layer is constructed. This information is necessary to
                differentiate the parameters of different layers.
        Returns:
            encoder_outputs: Output of LSTM, a 3-D tensor of shape TxBxH.
            final_state: Final hidden state of LSTM.
        """
        params = self.params
        with variable_scope.variable_scope("RNNLayer%d" % (layer_depth),
                                           initializer=tf.random_uniform_initializer(-0.075, 0.075)):
            # Check if the encoder needs to be bidirectional or not.
            if params.bi_dir:
                (encoder_output_fw, encoder_output_bw), _ = \
                    rnn.bidirectional_dynamic_rnn(
                        self.cell, self.cell, encoder_inputs,
                        sequence_length=seq_len, dtype=tf.float32,
                        time_major=True)
                # Concatenate the output of forward and backward layer
                encoder_outputs = tf.concat([encoder_output_fw,
                                             encoder_output_bw], 2)
            else:
                encoder_outputs, _ = rnn.dynamic_rnn(
                    self.cell, encoder_inputs, sequence_length=seq_len,
                    dtype=tf.float32, time_major=True)

            return encoder_outputs


    def _get_pyramid_input(self, input_tens, seq_len):
        """
        Assumes batch major input tensor - input_tens
        """
        params = self.params
        max_seq_len = tf.reduce_max(seq_len)
        check_rem = tf.cast(tf.mod(max_seq_len, params.skip_step), tf.int32)

        feat_size = input_tens.get_shape()[2].value

        div_input_tens = tf.cond(
            tf.cast(check_rem, tf.bool), ##Odd or even length
            lambda: tf.identity(
                tf.concat([input_tens,
                           tf.zeros([tf.shape(input_tens)[0],
                                     params.skip_step-check_rem, feat_size])], 1)),
            lambda: tf.identity(input_tens))

        output_tens = tf.reshape(
            div_input_tens, [tf.shape(div_input_tens)[0],
                             tf.cast(tf.shape(div_input_tens)[1]/params.skip_step, tf.int32),
                             feat_size * params.skip_step])
        # Get the ceil division since we pad it with 0s
        seq_len = tf.to_int32(tf.ceil(
            tf.truediv(seq_len, tf.cast(params.skip_step, dtype=tf.int64))))
        return output_tens, seq_len


    def __call__(self, encoder_input, seq_len, num_layers):
        """Run the encoder on gives input.

        Args:
            encoder_inp: Input IDs that are time major i.e. TxB. These IDs are
                first passed through embedding layer before feeding to first
                LSTM layer.
            seq_len: Actual length of input time sequences.
        Returns:
            attention_states: Final encoder output for every input timestep.
                This tensor is used by attention-enabled decoders.
        """
        params = self.params
        with variable_scope.variable_scope(
                "encoder", initializer=tf.random_uniform_initializer(-0.1, 0.1)):
            attention_states = {}
            time_major_states = {}
            seq_len_inps = {}
            max_depth = 0

            for task, num_layer in num_layers.items():
                if task == "state":
                    time_major_states[num_layer] = None
                else:
                    attention_states[num_layer] = None
                max_depth = max(max_depth, num_layers[task])

            resolution_fac = params.initial_res_fac  # Term to maintain time-resolution factor
            for i in xrange(max_depth):
                layer_depth = i+1
                # Transpose the input into time major input
                encoder_output = self._layer_encoder_input(tf.transpose(encoder_input, [1, 0, 2]),
                                                           seq_len, layer_depth)

                if time_major_states.has_key(layer_depth):
                    time_major_states[layer_depth] = encoder_output

                # Make the encoder output batch major
                encoder_output = tf.transpose(encoder_output, [1, 0, 2])

                if attention_states.has_key(layer_depth):
                    attention_states[layer_depth] = encoder_output

                seq_len_inps[layer_depth] = seq_len

                # For every character there are rougly 8 frames
                if params.skip_step > 1 and i != (max_depth-1) and resolution_fac <= 8:
                    encoder_input, seq_len = self._get_pyramid_input(
                        encoder_output, seq_len)
                    resolution_fac *= 2
                else:
                    encoder_input = encoder_output

            return attention_states, time_major_states, seq_len_inps

#    def encode_cudnn(self, encoder_inp):
#        """Use the Cudnn RNN encoder.
#
#        Args:
#            encoder_inp: Input vectors that are time major - TxBxH
#
#        Returns:
#            encoder_outputs: Since with Cudnn RNN it makes more sense to just
#                run a monolithic multilayered RNN, we just return the output
#                of last layer.
#        """

