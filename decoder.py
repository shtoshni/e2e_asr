"""Abstract decoder class for seq2seq model. Doesn't implement actual decoding.

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


class Decoder(BaseParams):
    """Base class for decoder in encoder-decoder framework."""

    @classmethod
    def class_params(cls):
        """Decoder class parameters."""
        params = Bunch()
        params['out_prob'] = 0.9
        params['hidden_size'] = 256
        params['num_layers_dec'] = 1
        params['emb_size'] = 256
        params['vocab_size'] = 1000
        params['samp_prob'] = 0.1
        params['max_output'] = 400
        params['use_lstm'] = False

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

        if self.isTraining and (params.samp_prob > 0.0):
            self.params.isSampling = True

    def get_cell(self):
        """Create the LSTM cell used by decoder."""
        params = self.params
        def single_cell():
            """Create a single RNN cell."""
            if params.use_lstm:
                cell = tf.nn.rnn_cell.BasicLSTMCell(params.hidden_size)
            else:
                cell = tf.nn.rnn_cell.GRUCell(params.hidden_size)
            if self.isTraining:
                # During training we use a dropout wrapper
                cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell, output_keep_prob=params.out_prob)
            return cell

        if params.num_layers_dec > 1:
            # If RNN is stacked then we use MultiRNNCell class
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell() for _ in xrange(params.num_layers_dec)])
        else:
            cell = single_cell()

        # Use the OutputProjectionWrapper to project cell output to output
        # vocab size. This projection is fine for a small vocabulary output
        # but would be bad for large vocabulary output spaces.
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, params.vocab_size)
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
        if self.isTraining:
            if params.isSampling:
                # This loop function samples the output from the posterior
                # and embeds this output.
                loop_function = self._sample_argmax(embedding)
                print ("Scheduled sampling!")
            else:
                loop_function = None
        else:
            # Get the loop function that would embed the maximum posterior
            # symbol. This funtion is used during decoding in RNNs
            loop_function = self._get_argmax(embedding)

        return (embedded_inp, loop_function)

    @abc.abstractmethod
    def __call__(self, decoder_inp, seq_len, encoder_hidden_states,
                 seq_len_inp):
        """Abstract method that needs to be extended by Inheritor classes.

        Args:
            decoder_inp: Time major decoder IDs, TxB that contain ground truth
                during training and are dummy value holders at test time.
            seq_len: Output sequence length for each input in minibatch.
                Useful to limit the computation to the max output length in
                a minibatch.
            encoder_hidden_states: Batch major output, BxTxH of encoder RNN.
                Useful with attention-enabled decoders.
            final_state: Final hidden state of encoder RNN. Useful for
                initializing decoder RNN.
            seq_len_inp: Useful with attention-enabled decoders to mask the
                outputs corresponding to padding symbols.
        Returns:
            outputs: Time major output, TxBx|V|, of decoder RNN.
        """
        pass

    def _get_argmax(self, embedding):
        """Return a function that returns the previous output with max prob.

        Args:
            embedding : Embedding matrix for embedding the symbol
        Returns:
            loop_function: A function that returns the embedded output symbol
                with maximum probability (logit score).
        """
        def loop_function(logits):
            max_symb = tf.argmax(logits, 1)
            emb_symb = tf.nn.embedding_lookup(embedding, max_symb)
            return emb_symb

        return loop_function

    def _sample_argmax(self, embedding):
        """Return a function that samples from posterior over previous output.

        Args:
            embedding : Embedding matrix for embedding the symbol
        Returns:
            loop_function: A function that samples the output symbol from
            posterior and embeds the sampled symbol.
        """
        def loop_function(prev):
            """The closure function returned by outer function.

            Args:
                prev: logit score for previous step output
            Returns:
                emb_prev: The embedding of output symbol sampled from
                    posterior over previous output.
            """
            # tf.multinomial performs sampling given the logit scores
            # Reshaping is required to remove the extra dimension introduced
            # by sampling for a batch size of 1.
            prev_symbol = tf.reshape(tf.multinomial(prev, 1), [-1])
            emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
            return emb_prev

        return loop_function
