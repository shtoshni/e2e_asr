class BeamEntry(object):
    """Beam entry class that stores info relevant for performing beam search."""

    def __init__(self, index_seq, dec_state, context_vec):
        self.index_seq = index_seq
        self.dec_state = dec_state
        self.context_vec = context_vec

    def get_last_output(self):
        return self.index_seq[-1]

    def get_index_seq(self):
        return self.index_seq

    def get_dec_state(self):
        return self.dec_state

    def get_context_vec(self):
        return self.context_vec
