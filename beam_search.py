from bunch import Bunch

import numpy as np

import tf_utils
import data_utils

from beam_entry import BeamEntry
from copy import deepcopy

class BeamSearch(object):

    def __init__(self, ckpt_path, rev_vocab, beam_size=4):
        """Initialize the model."""
        self.params = self.map_variables(self.get_model_params(ckpt_path))
        self.rev_vocab = rev_vocab
        self.beam_size = beam_size

    def get_model_params(self, ckpt_path):
        """Loads the decoder params"""
        return tf_utils.get_matching_variables("rnn_decoder_char", ckpt_path)


    def map_variables(self, var_dict):
        """Map loaded tensors from names to variables."""
        params = Bunch()
        params.lstm_w = var_dict[
            "model/rnn_decoder_char/rnn/basic_lstm_cell/kernel"]
        params.lstm_b = var_dict[
            "model/rnn_decoder_char/rnn/basic_lstm_cell/bias"]

        params.inp_w = var_dict[
            "model/rnn_decoder_char/rnn/InputProjection/kernel"]
        params.inp_b = var_dict[
            "model/rnn_decoder_char/rnn/InputProjection/bias"]

        params.attn_proj_w = var_dict[
            "model/rnn_decoder_char/rnn/AttnOutputProjection/kernel"]
        params.attn_proj_b = var_dict[
            "model/rnn_decoder_char/rnn/AttnOutputProjection/bias"]

        params.attn_dec_w = var_dict[
            "model/rnn_decoder_char/rnn/Attention/kernel"]
        params.attn_dec_b = var_dict[
            "model/rnn_decoder_char/rnn/Attention/bias"]

        params.attn_enc_w = np.squeeze(var_dict["model/rnn_decoder_char/AttnW"])

        params.attn_v = var_dict["model/rnn_decoder_char/AttnV"]

        params.embedding = var_dict["model/rnn_decoder_char/decoder/embedding"]
        return params


    def calc_attention(self, encoder_hidden_states):
        params = self.params
        if len(encoder_hidden_states.shape) == 3:
            # Squeeze the first dimension
            encoder_hidden_states = np.squeeze(encoder_hidden_states, axis=0)

        # T x Attn_vec_size
        attn_enc_term = np.matmul(encoder_hidden_states, params.attn_enc_w)


        def attention(dec_state):
            attn_dec_term = (np.matmul(dec_state, params.attn_dec_w) +
                             params.attn_dec_b)  # T x A
            attn_sum = np.tanh(attn_enc_term + attn_dec_term) # T x A
            attn_logits = np.squeeze(np.matmul(attn_sum, params.attn_v))  # T
            attn_probs = self.softmax(attn_logits)

            context_vec = np.matmul(attn_probs, encoder_hidden_states)
            return context_vec

        return attention

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference

    def dec_lstm_func(self):
        params = self.params
        lstm_w = params.lstm_w
        lstm_b = params.lstm_b

        def call_lstm(x, lstm_state):
            c, h = lstm_state
            x_h = np.concatenate((x, h), axis=0)
            i, j, f, o = np.split(
                np.matmul(x_h, lstm_w) + lstm_b, 4)
            f_gate = self.sigmoid(f + 1)   # 1 for forget bias
            new_c = (np.multiply(c, f_gate) +
                     np.multiply(self.sigmoid(i), np.tanh(j)))
            new_h = np.multiply(self.sigmoid(o), np.tanh(new_c))
            return (new_c, new_h)

        return call_lstm

    def top_k_setup(self, encoder_hidden_states):
        params = self.params
        dec_lstm_call = self.dec_lstm_func()
        attention_call = self.calc_attention(encoder_hidden_states)

        def get_top_k(x, lstm_state, beam_size=self.beam_size):
            lstm_state = dec_lstm_call(x, lstm_state)
            attn_c = attention_call(lstm_state[0])
            attn_dec_state = np.concatenate((lstm_state[0], attn_c), axis=0)

            output_probs = self.softmax(np.matmul(attn_dec_state, params.attn_proj_w) + params.attn_proj_b)
            top_k_indices = np.argpartition(output_probs, -beam_size)[-beam_size:]

            # Return indices, their score, and the lstm state
            return (top_k_indices, np.log(output_probs[top_k_indices]), lstm_state, attn_c)

        return get_top_k


    def __call__(self, encoder_hidden_states):
        """Beam search for batch_size=1"""
        params = self.params

        get_top_k_fn = self.top_k_setup(encoder_hidden_states)

        x = params.embedding[data_utils.GO_ID]
        h_size = params.lstm_w.shape[1]/4

        lstm_c, lstm_h = np.zeros(h_size), np.zeros(h_size)

        # Maintain a tuple of (output_indices, score, encountered EOS?)
        output_list = []
        final_output_list = []
        k = self.beam_size
        step_count = 1

        # Run step 0 separately
        top_k_indices, top_k_scores, lstm_state, attn_state = get_top_k_fn(x, (lstm_c, lstm_h), beam_size=k)
        for idx in xrange(top_k_indices.shape[0]):
            output_tuple = (BeamEntry([top_k_indices[idx]], lstm_state, attn_state), top_k_scores[idx])
            if top_k_indices[idx] == data_utils.EOS_ID:
                final_output_list.append(output_tuple)
                k -= 1
            else:
                output_list.append(output_tuple)

        while step_count < 120 and k > 0:
            next_contender = []
            for candidate, cand_score in output_list:
                simple_input = params.embedding[candidate.last_output()]
                concat_input = np.concatenate((simple_input, candidate.get_attn_state()), axis=0)
                x = np.matmul(concat_input, params.inp_w) + params.inp_b
                top_k_indices, top_k_scores, lstm_state, attn_state =\
                    get_top_k_fn(x, candidate.get_state(), beam_size=k)

                for idx in xrange(k):
                    new_cand_score = cand_score + top_k_scores[idx]
                    new_index_seq = candidate.index_seq + [top_k_indices[idx]]
                    new_candidate = BeamEntry(new_index_seq, lstm_state, attn_state)
                    next_contender.append((new_candidate, new_cand_score))

            output_list = sorted(next_contender, key=lambda cand_tuple: cand_tuple[1])[-k:]
            for idx in reversed(xrange(k)):
                if output_list[idx][0].last_output() == data_utils.EOS_ID:
                    final_output_list.append(output_list.pop(idx))
                    k -= 1
            step_count += 1

        final_output_list += output_list

        best_output = max(final_output_list, key=lambda output_tuple: output_tuple[1])
        output_seq = best_output[0].get_index_seq()
        return np.stack(output_seq, axis=0)
