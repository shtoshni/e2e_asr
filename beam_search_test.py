from beam_search import BeamSearch

import data_utils

class BeamSearchTest(object):

    ckpt_file = ("/share/data/speech/shtoshni/research/asr_multi/models/"
                  "skip_2_lstm_lm_prob_0.75_run_id_205/asr.ckpt-39000")
    vocab_file = ("/share/data/speech/shtoshni/research/datasets/"
                  "asr_swbd/lang/vocab/char.vocab")

    def __init__(self):
        self.rev_vocab = self.load_char_vocab()
        self.beam_search = BeamSearch(self.ckpt_file, self.rev_vocab)

    def load_char_vocab(self):
        _, rev_char_vocab = data_utils.initialize_vocabulary(self.vocab_file)
        return rev_char_vocab

    def test_param_load(self):
        """Test file loading."""
        for param_name in self.beam_search.params:
            param_shape = self.beam_search.params[param_name].shape
            print (param_name + ": " + str(param_shape))


if __name__=="__main__":
    beam_search_tester = BeamSearchTest()
    beam_search_tester.test_param_load()
