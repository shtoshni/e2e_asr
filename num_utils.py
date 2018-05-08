"""Implementation of some useful numerical functions using numpy."""

import cupy as np


def sigmoid(x):
    """Compute the sigmoid function."""
    return 1/(1 + np.exp(-x))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference
