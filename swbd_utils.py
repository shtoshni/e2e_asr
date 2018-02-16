import re
import tensorflow as tf

import data_utils


def reverse_swbd_normalizer():
    swbd_dict = {"!": "[laughter]",
                 "@": "[noise]",
                 "#": "[vocalized-noise]"}

    def normalizer(text):
        # Create a regex for match
        regex = re.compile("(%s)" % "|".join(map(re.escape, swbd_dict.keys())))
        # For each match, look-up corresponding value in dictionary
        return regex.sub(lambda match: swbd_dict[match.string[match.start() : match.end()]], text)

    return normalizer


