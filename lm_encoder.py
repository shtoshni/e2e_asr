"""LM Encoder class for using the decoder as LM.

Author: Shubham Toshniwal
Contact: shtoshni@ttic.edu
Date: February, 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from decoder import Decoder
from base_params import BaseParams


class LMEncoder(Decoder, BaseParams):
    """Implements the LM model using decoder params."""

    @classmethod
    def class_params(cls):
        """Defines params of the class."""
        params = super(LMEncoder, cls).class_params()
        params['encoder_hidden_size'] = 256
        return params

    def __init__(self, params=None):
        """Initializer."""
        super(LMEncoder, self).__init__(isTraining=True, params=params)
        # No output projection required in attention decoder
        self.cell = self.get_cell()

    def get_state(self, state):
        """Get the state while handling multiple layer and different cell cases."""
        params = self.params
        if params.num_layers_dec > 1:
            state = state[-1]
        if params.use_lstm:
            state = state.c

        return state

    def __call__(self, decoder_inp, seq_len):
        # First prepare the decoder input - Embed the input and obtain the
        # relevant loop function
        params = self.params
        scope = "rnn_decoder_char"

        with tf.variable_scope(scope, reuse=True):
            decoder_inputs, loop_function = self.prepare_decoder_input(decoder_inp)

        # TensorArray is used to do dynamic looping over decoder input
        inputs_ta = tf.TensorArray(size=params.max_output,
                                   dtype=tf.float32)
        inputs_ta = inputs_ta.unstack(decoder_inputs)

        batch_size = tf.shape(decoder_inputs)[1]
        emb_size = decoder_inputs.get_shape()[2].value

        batch_attn_size = tf.stack([batch_size, params.encoder_hidden_size])
        zero_attn = tf.zeros(batch_attn_size, dtype=tf.float32)

        with tf.variable_scope(scope, reuse=True):
            def raw_loop_function(time, cell_output, state, loop_state):
                # If loop_function is set, we use it instead of decoder_inputs.
                elements_finished = (time >= tf.cast(seq_len, tf.int32))
                finished = tf.reduce_all(elements_finished)


                if cell_output is None:
                    next_state = self.cell.zero_state(batch_size, dtype=tf.float32)
                    output = None
                    loop_state = None
                    next_input = inputs_ta.read(time)
                else:
                    next_state = state
                    with tf.variable_scope("AttnOutputProjection"):
                        output = _linear([self.get_state(state), zero_attn],
                                         self.cell.output_size, True)

                    if loop_function is not None:
                        random_prob = tf.random_uniform([])
                        simple_input = tf.cond(
                            finished, lambda: tf.zeros([batch_size, emb_size], dtype=tf.float32),
                            lambda: tf.cond(tf.less(random_prob, 1 - params.samp_prob),
                                            lambda: inputs_ta.read(time),
                                            lambda: loop_function(output))
                            )
                    else:
                        simple_input = tf.cond(
                            finished, lambda: tf.zeros([batch_size, emb_size], dtype=tf.float32),
                            lambda: inputs_ta.read(time)
                            )

                    # Merge input and previous attentions into one vector of the right size.
                    input_size = simple_input.get_shape().with_rank(2)[1]
                    if input_size.value is None:
                        raise ValueError("Could not infer input size from input")
                    with tf.variable_scope("InputProjection"):
                        next_input = _linear([simple_input, zero_attn], input_size, True)

                return (elements_finished, next_input, next_state, output, loop_state)

            # outputs is a TensorArray with T=max(sequence_length) entries
            # of shape Bx|V|
            outputs, _, _ = tf.nn.raw_rnn(self.cell, raw_loop_function)

        # Concatenate the output across timesteps to get a tensor of TxBx|V|
        # shape
        outputs = outputs.concat()

        return outputs
