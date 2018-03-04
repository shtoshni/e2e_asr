"""Implements the Basic LSTM cell of tensorflow required for beam search in numpy."""

import numpy as np

from num_utils import sigmoid

class BasicLSTM(object):
    """Implementation of the basic LSTM cell from tensorflow."""

    def __init__(self, weight, bias):
        self.lstm_w = weight
        self.lstm_b = bias

    def __call__(self, x, lstm_state):
        c, h = lstm_state
        x_h = np.concatenate((x, h), axis=0)
        i, j, f, o = np.split(
            np.matmul(x_h, self.lstm_w) + self.lstm_b, 4)
        f_gate = sigmoid(f + 1)   # 1 for forget bias
        new_c = (np.multiply(c, f_gate) +
                 np.multiply(sigmoid(i), np.tanh(j)))
        new_h = np.multiply(sigmoid(o), np.tanh(new_c))
        return (new_c, new_h)
