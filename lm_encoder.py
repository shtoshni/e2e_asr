from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.rnn as rnn_cell

from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from decoder import Decoder


class LM(Decoder):
    """Implements the LM model using decoder params."""

    @classmethod
    def class_params(cls):
        """Defines params of the class."""
        params = super(LM, cls).class_params()
        params['encoder_hidden_size'] = 256
        return params

    def __init__(self, params=None, scope=None):
        """Initializer."""
        super(LM, self).__init__(params)
        # No output projection required in attention decoder
        self.scope = scope
        self.cell = self.get_cell()

    def __call__(self, decoder_inp, seq_len):
        # First prepare the decoder input - Embed the input and obtain the
        # relevant loop function
        params = self.params
        scope = "rnn_decoder" + ("" if self.scope is None else "_" + self.scope)

        with variable_scope.variable_scope(scope):
            decoder_inputs, loop_function = self.prepare_decoder_input(decoder_inp)

        # TensorArray is used to do dynamic looping over decoder input
        inputs_ta = tf.TensorArray(size=params.max_output,
                                   dtype=tf.float32)
        inputs_ta = inputs_ta.unstack(decoder_inputs)

        batch_size = tf.shape(decoder_inputs)[1]
        emb_size = decoder_inputs.get_shape()[2].value

        batch_attn_size = array_ops.stack([batch_size, params.encoder_hidden_size])
        zero_attn = array_ops.zeros(batch_attn_size, dtype=dtypes.float32)

        with variable_scope.variable_scope(scope):
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
                    with variable_scope.variable_scope("AttnOutputProjection"):
                        if params.use_lstm:
                            output = _linear([state.c, zero_attn],
                                             self.cell.output_size, True)
                        else:
                            output = _linear([state, zero_attn],
                                             self.cell.output_size, True)


                    if loop_function is not None:
                        print("Scheduled Sampling will be done")
                        random_prob = tf.random_uniform([])
                        simple_input = tf.cond(finished,
                            lambda: tf.zeros([batch_size, emb_size], dtype=tf.float32),
                            lambda: tf.cond(tf.less(random_prob, 0.9),
                                lambda: inputs_ta.read(time),
                                lambda: loop_function(output))
                            )
                    else:
                        simple_input = tf.cond(finished,
                            lambda: tf.zeros([batch_size, emb_size], dtype=tf.float32),
                            lambda: inputs_ta.read(time)
                            )

                    # Merge input and previous attentions into one vector of the right size.
                    input_size = simple_input.get_shape().with_rank(2)[1]
                    if input_size.value is None:
                        raise ValueError("Could not infer input size from input")
                    with variable_scope.variable_scope("InputProjection"):
                        next_input = _linear([simple_input, zero_attn], input_size, True)

                return (elements_finished, next_input, next_state, output, loop_state)

            # outputs is a TensorArray with T=max(sequence_length) entries
            # of shape Bx|V|
            outputs, state, _ = rnn.raw_rnn(self.cell, raw_loop_function)

        # Concatenate the output across timesteps to get a tensor of TxBx|V|
        # shape
        outputs = outputs.concat()

        return outputs
