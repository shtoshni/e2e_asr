
class BeamEntry(object):
    """Beam entry class that stores info relevant for performing beam search."""

    def __init__(self, index_seq, lstm_state, attn_state):
        self.index_seq = index_seq
        self.lstm_state = lstm_state
        self.attn_state = attn_state

    def last_output(self):
        return self.index_seq[-1]

    def get_index_seq(self):
        return self.index_seq

    def update_seq(self, elem):
        self.index_seq.append(elem)

    def get_state(self):
        return self.lstm_state

    def set_state(self, lstm_state):
        self.lstm_state = lstm_state

    def get_attn_state(self):
        return self.attn_state

    def set_attn_state(self, attn_state):
        self.attn_state = attn_state
