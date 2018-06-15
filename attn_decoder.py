"""Attention-enabled decoder class of seq2seq model.

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


class AttnDecoder(Decoder, BaseParams):
    """Implements the attention decoder of encoder-decoder framework."""

    @classmethod
    def class_params(cls):
        """Defines params of the class."""
        params = super(AttnDecoder, cls).class_params()
        params['tie_embedding'] = False
        params['attention_vec_size'] = 128
        params['lm_hidden_size'] = 256
        params['ind_softmax'] = False
        return params

    def __init__(self, isTraining, params=None, scope=None):
        """Initializer."""
        super(AttnDecoder, self).__init__(isTraining=isTraining, params=params)
        # No output projection required in attention decoder
        self.scope = scope
        self.cell = self.get_cell()

    def __call__(self, decoder_inp, seq_len,
                 encoder_hidden_states, seq_len_inp):
        # First prepare the decoder input - Embed the input and obtain the
        # relevant loop function
        params = self.params

        scope = "rnn_decoder" + ("" if self.scope is None else "_" + self.scope)

        with tf.variable_scope(scope):
            if params.tie_embedding:
                embedding, decoder_inputs, loop_function = self.prepare_decoder_input(
                    decoder_inp, get_embedding=True)
            else:
                decoder_inputs, loop_function = self.prepare_decoder_input(decoder_inp)
            lm_cell = self.get_cell(hidden_size=params.lm_hidden_size)

        # TensorArray is used to do dynamic looping over decoder input
        inputs_ta = tf.TensorArray(size=params.max_output,
                                   dtype=tf.float32)
        inputs_ta = inputs_ta.unstack(decoder_inputs)

        batch_size = tf.shape(decoder_inputs)[1]
        attn_length = tf.shape(encoder_hidden_states)[1]
        emb_size = decoder_inputs.get_shape()[2].value
        attn_size = encoder_hidden_states.get_shape()[2].value

        # Attention variables
        attn_mask = tf.sequence_mask(tf.cast(seq_len_inp, tf.int32), dtype=tf.float32)

        batch_attn_size = tf.stack([batch_size, attn_size])
        attn = tf.zeros(batch_attn_size, dtype=tf.float32)
        batch_alpha_size = tf.stack([batch_size, attn_length, 1, 1])
        alpha = tf.zeros(batch_alpha_size, dtype=tf.float32)


        with tf.variable_scope(scope):
            # Calculate the W*h_enc component
            hidden = tf.expand_dims(encoder_hidden_states, 2)
            W_attn = tf.get_variable(
                "AttnW", [1, 1, attn_size, params.attention_vec_size])
            hidden_features = tf.nn.conv2d(hidden, W_attn, [1, 1, 1, 1], "SAME")
            v = tf.get_variable("AttnV", [params.attention_vec_size])

            def raw_loop_function(time, cell_output, state, loop_state):
                def attention(query, prev_alpha):
                    """Put attention masks on hidden using hidden_features and query."""
                    with tf.variable_scope("Attention"):
                        y = _linear(query, params.attention_vec_size, True)
                        y = tf.reshape(y, [-1, 1, 1, params.attention_vec_size])
                        s = tf.reduce_sum(
                            v * tf.tanh(hidden_features + y), [2, 3])

                        alpha = tf.nn.softmax(s) * attn_mask
                        sum_vec = tf.reduce_sum(alpha, reduction_indices=[1], keepdims=True)
                        norm_term = tf.tile(sum_vec, tf.stack([1, tf.shape(alpha)[1]]))
                        alpha = alpha / norm_term

                        alpha = tf.expand_dims(alpha, 2)
                        alpha = tf.expand_dims(alpha, 3)
                        context_vec = tf.reduce_sum(alpha * hidden, [1, 2])
                    return tuple([context_vec, alpha])

                # If loop_function is set, we use it instead of decoder_inputs.
                elements_finished = (time >= tf.cast(seq_len, tf.int32))
                finished = tf.reduce_all(elements_finished)


                if cell_output is None:
                    next_state = self.cell.zero_state(batch_size, dtype=tf.float32)

                    # This output is not used but is just used to tell the shape
                    # without the batch dimension
                    # Check here - https://www.tensorflow.org/api_docs/python/tf/nn/raw_rnn
                    output = tf.zeros((self.params.vocab_size))
                    lm_input = inputs_ta.read(time)
                    attn_state = tuple([attn, alpha])
                    lm_state = lm_cell.zero_state(batch_size, dtype=tf.float32)
                else:
                    next_state = state
                    #loop_state = attention(cell_output, loop_state[1])
                    lm_state, attn_state = loop_state
                    attn_state = attention(self.get_state(state), attn_state[1])

                    with tf.variable_scope("AttnProjection"):
                        proj_output = _linear([self.get_state(state), attn_state[0]],
                                              self.params.hidden_size_dec, True)
                    if params.tie_embedding:
                        output = (tf.matmul(proj_output, tf.transpose(embedding)) +
                                  tf.get_variable("output_bias", [params.vocab_size]))
                    else:
                        if params.ind_softmax:
                            # Don't share parameters with LM model
                            with tf.variable_scope("OutputProjection2"):
                                output = _linear([proj_output], self.params.vocab_size, True)
                        else:
                            with tf.variable_scope("OutputProjection"):
                                output = _linear([proj_output], self.params.vocab_size, True)


                    if not self.isTraining:
                        lm_input = loop_function(output)
                    else:
                        if loop_function is not None:
                            random_prob = tf.random_uniform([])
                            lm_input = tf.cond(
                                finished,
                                lambda: tf.zeros([batch_size, emb_size], dtype=tf.float32),
                                lambda: tf.cond(tf.less(random_prob, 1 - params.samp_prob),
                                                lambda: inputs_ta.read(time),
                                                lambda: loop_function(output))
                            )
                        else:
                            lm_input = tf.cond(
                                finished,
                                lambda: tf.zeros([batch_size, emb_size], dtype=tf.float32),
                                lambda: inputs_ta.read(time)
                            )

                # Common calculations
                with tf.variable_scope("lm"):
                    lm_output, next_lm_state = lm_cell(lm_input, lm_state)
                if params.lm_hidden_size != params.hidden_size_dec:
                    with tf.variable_scope("SimpleProjection", reuse=tf.AUTO_REUSE):
                        lm_output = _linear([lm_output], params.hidden_size_dec, True)

                # Merge input and previous attentions into one vector of the right size.
                input_size = lm_input.get_shape().with_rank(2)[1]
                if input_size.value is None:
                    raise ValueError("Could not infer input size from input")
                with tf.variable_scope("InputProjection", reuse=tf.AUTO_REUSE):
                    next_input = _linear([lm_output, attn_state[0]], input_size, True)

                loop_state = tuple([next_lm_state, attn_state])

                return (elements_finished, next_input, next_state, output, loop_state)

            # outputs is a TensorArray with T=max(sequence_length) entries
            # of shape Bx|V|
            outputs, state, _ = tf.nn.raw_rnn(self.cell, raw_loop_function)

        # Concatenate the output across timesteps to get a tensor of TxBx|V|
        # shape
        outputs = outputs.concat()

        return outputs

    @classmethod
    def add_parse_options(cls, parser):
        """Add decoder specific arguments."""
        # Decoder params
        super(AttnDecoder, cls).add_parse_options(parser)
        parser.add_argument("-tie_emb", "--tie_embedding", default=False, action="store_true",
                            help="Tie output projection to input embedding")
        parser.add_argument("-samp_prob", "--samp_prob", default=0.1, type=float,
                            help="Scheduled sampling probability")
        parser.add_argument("-attn_vec_size", "--attention_vec_size", default=128,
                            type=int, help="Attention vector size")
        parser.add_argument("-lm_hsize", "--lm_hidden_size", default=512,
                            type=int, help="Hidden Size of LM layer")
        parser.add_argument('-ind_softmax', "--ind_softmax", default=False,
                            action="store_true", help="Independent (from LM) softmax params")
