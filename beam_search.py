from bunch import Bunch
from copy import deepcopy

import numpy as np

import tf_utils
import data_utils

from beam_entry import BeamEntry
from num_utils import softmax
from basic_lstm import BasicLSTM
from base_params import BaseParams


class BeamSearch(BaseParams):
    """Implementation of beam search for the attention decoder assuming a
    batch size of 1."""

    @classmethod
    def class_params(cls):
        """Decoder class parameters."""
        params = Bunch()
        params['beam_size'] = 4
        params['lm_weight'] = 0.0
        params['lm_path'] = ""
        params['word_ins_penalty'] = 0.0

        return params

    def __init__(self, ckpt_path, rev_vocab, search_params=None):
        """Initialize the model."""
        self.dec_params = self.map_dec_variables(self.get_model_params(ckpt_path))
        self.rev_vocab = rev_vocab

        if search_params is None:
            self.search_params = self.class_params()
        else:
            self.search_params = search_params

        if self.search_params.lm_path is None or (self.search_params.lm_weight == 0.0):
            self.use_lm = False
            print ("No separate LM used")
        else:
            self.use_lm = True
            self.lm_params = self.map_lm_variables(
                self.get_model_params(self.search_params.lm_path))
        print ("Using a beam size of %d" %self.search_params.beam_size)

    def get_model_params(self, ckpt_path):
        """Loads the decoder params"""
        return tf_utils.get_matching_variables("rnn_decoder_char", ckpt_path)

    def map_dec_variables(self, var_dict):
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

    def map_lm_variables(self, var_dict):
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

        params.embedding = var_dict["model/rnn_decoder_char/decoder/embedding"]
        return params


    def calc_attention(self, encoder_hidden_states):
        """Context vector calculation function. Here the encoder's contribution
        to attention remains the same and can be computed earlier. We perform
        currying to return a function that takes as input just the decoder state."""

        params = self.dec_params
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
            attn_probs = softmax(attn_logits)

            context_vec = np.matmul(attn_probs, encoder_hidden_states)
            return context_vec

        return attention

    def top_k_setup_with_lm(self, encoder_hidden_states):
        params = self.dec_params
        lm_params = self.lm_params
        search_params = self.search_params

        # Set up decoder components
        dec_lstm = BasicLSTM(params.lstm_w, params.lstm_b)
        attention_call = self.calc_attention(encoder_hidden_states)

        # Set up LM components
        lm_lstm = BasicLSTM(lm_params.lstm_w, lm_params.lstm_b)
        # LM uses a zero attn vector
        zero_attn = np.zeros(encoder_hidden_states.shape[1])

        def get_top_k(x_dec, x_lm, state_list, beam_size=search_params.beam_size):
            dec_state, lm_state = state_list

            dec_state = dec_lstm(x_dec, dec_state)
            context_vec = attention_call(dec_state[0])
            context_dec_comb = np.concatenate((dec_state[0], context_vec), axis=0)
            output_dec_probs = softmax(np.matmul(context_dec_comb, params.attn_proj_w) +
                                       params.attn_proj_b)
            log_dec_probs = np.log(output_dec_probs)

            lm_state = lm_lstm(x_lm, lm_state)
            context_lm_comb = np.concatenate((lm_state[0], zero_attn), axis=0)
            output_lm_probs = softmax(np.matmul(context_lm_comb, lm_params.attn_proj_w) +
                                      lm_params.attn_proj_b)
            log_lm_probs = np.log(output_lm_probs)

            combined_log_probs = log_dec_probs + search_params.lm_weight * log_lm_probs

            top_k_indices = np.argpartition(combined_log_probs, -beam_size)[-beam_size:]

            # Return indices, their score, and the lstm state
            return (top_k_indices, combined_log_probs[top_k_indices], [dec_state,
                    lm_state], context_vec)

        return get_top_k


    def top_k_setup(self, encoder_hidden_states):
        params = self.dec_params
        search_params = self.search_params

        dec_lstm = BasicLSTM(params.lstm_w, params.lstm_b)
        attention_call = self.calc_attention(encoder_hidden_states)

        def get_top_k(x, state_list, beam_size=search_params.beam_size):
            dec_state = state_list[0]
            dec_state = dec_lstm(x, dec_state)
            context_vec = attention_call(dec_state[0])
            context_dec_comb = np.concatenate((dec_state[0], context_vec), axis=0)

            output_probs = softmax(np.matmul(context_dec_comb, params.attn_proj_w) +
                                   params.attn_proj_b)
            top_k_indices = np.argpartition(output_probs, -beam_size)[-beam_size:]

            # Return indices, their score, and the lstm state
            return (top_k_indices, np.log(output_probs[top_k_indices]), [dec_state], context_vec)

        return get_top_k


    def __call__(self, encoder_hidden_states):
        """Beam search for batch_size=1"""
        params = self.dec_params
        search_params = self.search_params

        if self.use_lm:
            lm_params = self.lm_params

        if self.use_lm:
            get_top_k_fn = self.top_k_setup_with_lm(encoder_hidden_states)
        else:
            get_top_k_fn = self.top_k_setup(encoder_hidden_states)

        x = params.embedding[data_utils.GO_ID]
        if self.use_lm:
            x_lm = lm_params.embedding[data_utils.GO_ID]
        h_size = params.lstm_w.shape[1]/4

        zero_state = (np.zeros(h_size), np.zeros(h_size))
        zero_attn = np.zeros(encoder_hidden_states.shape[1])

        # Maintain a tuple of (output_indices, score, encountered EOS?)
        output_list = []
        final_output_list = []
        k = search_params.beam_size  # Represents the current beam size
        step_count = 0

        # Run step 0 separately
        if self.use_lm:
            top_k_indices, top_k_scores, state_list, attn_state =\
                get_top_k_fn(x, x_lm, [zero_state, zero_state], beam_size=k)
        else:
            top_k_indices, top_k_scores, state_list, attn_state =\
                get_top_k_fn(x, [zero_state], beam_size=k)
        for idx in xrange(top_k_indices.shape[0]):
            output_tuple = (BeamEntry([top_k_indices[idx]], state_list, attn_state), top_k_scores[idx])
            if top_k_indices[idx] == data_utils.EOS_ID:
                final_output_list.append(output_tuple)
                # Decrease the beam size once EOS is encountered
                k -= 1
            else:
                output_list.append(output_tuple)

        step_count += 1
        while step_count < 120 and k > 0:
            # These lists store the states obtained by running the decoder
            # for 1 more step with the previous outputs of the beam
            next_dec_states = []
            next_context_vecs = []

            score_list = []
            index_list = []
            for candidate, cand_score in output_list:
                simple_input = params.embedding[candidate.get_last_output()]
                concat_input = np.concatenate((simple_input, candidate.get_context_vec()),
                                              axis=0)
                x = np.matmul(concat_input, params.inp_w) + params.inp_b

                if self.use_lm:
                    lm_emb_input = lm_params.embedding[candidate.get_last_output()]
                    concat_lm = np.concatenate((lm_emb_input, zero_attn), axis=0)
                    x_lm = np.matmul(concat_lm, lm_params.inp_w) + lm_params.inp_b

                    top_k_indices, top_k_scores, state_list, context_vec =\
                        get_top_k_fn(x, x_lm, candidate.get_dec_state(), beam_size=k)
                else:
                    top_k_indices, top_k_scores, state_list, context_vec =\
                        get_top_k_fn(x, candidate.get_dec_state(), beam_size=k)

                next_dec_states.append(state_list)
                next_context_vecs.append(context_vec)

                index_list.append(top_k_indices)
                score_list.append(top_k_scores + cand_score)

            # Score of all k**2 continuations
            all_scores = np.concatenate(score_list, axis=0)
            # All k**2 continuations
            all_indices = np.concatenate(index_list, axis=0)

            # Find the top indices among the k^^2 entries
            top_k_indices = np.argpartition(all_scores, -k)[-k:]
            next_k_indices = all_indices[top_k_indices]
            top_k_scores = all_scores[top_k_indices]
            # The original candidate indices can be found by dividing by k.
            # Because the indices are of the form - i * k + j, where i
            # represents the ith output and j represents the jth top index for i
            orig_cand_indices = np.divide(top_k_indices, k, dtype=np.int32)

            new_output_list = []

            for idx in xrange(k):
                orig_cand_idx = orig_cand_indices[idx]
                # BeamEntry of the original candidate
                orig_cand = output_list[orig_cand_idx][0]
                next_elem = next_k_indices[idx]
                # Add the next index to the original sequence
                new_index_seq = orig_cand.get_index_seq() + [next_elem]
                dec_state = next_dec_states[orig_cand_idx]
                context_vec = next_context_vecs[orig_cand_idx]

                output_tuple = (BeamEntry(new_index_seq, dec_state, context_vec),
                                top_k_scores[idx])
                if next_elem == data_utils.EOS_ID:
                    # This sequence is finished. Put the output on the final list
                    # and reduce beam size
                    final_output_list.append(output_tuple)
                    k -= 1
                else:
                    new_output_list.append(output_tuple)

            output_list = new_output_list
            step_count += 1

        final_output_list += output_list

        best_output = max(final_output_list, key=lambda output_tuple: output_tuple[1])
        output_seq = best_output[0].get_index_seq()
        return np.stack(output_seq, axis=0)

    @classmethod
    def add_parse_options(cls, parser):
        """Add beam search specific arguments."""
        # Decoder params
        parser.add_argument("-beam_size", default=4, type=int, help="Beam size")
        parser.add_argument("-lm_weight", default=0.0, type=float, help="LM weight in decoding")
        parser.add_argument("-lm_path", default="/share/data/speech/shtoshni/research/asr_multi/"
                            "code/lm/models/pretrain_simple_lm/lm.ckpt-222000", type=str,
                            help="LM ckpt path")
        parser.add_argument("-word_ins_penalty", default=0.0, type=float,
                            help="Word insertion penalty")
