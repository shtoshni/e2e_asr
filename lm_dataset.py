"""Class for working with Language modeling datasets."""

import tensorflow as tf

class LMDataset(object):
    """Dataset class for language model dataset via TFRecords."""

    def __init__(self, filenames, batch_size):
        self.batch_size = batch_size
        self.data_set, self.data_iter = self.create_iterator(filenames)

    def get_instance(self, proto):
        """Parse the proto to prepare instance."""
        context_features = {
            "cint_len": tf.FixedLenFeature([], tf.int64),
        }
        sequence_features = {
            "cint": tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64),
        }
        # parse a sequence example given the above instructions on the structure
        context, sequence = tf.parse_single_sequence_example(
            serialized=proto,
            context_features=context_features,
            sequence_features=sequence_features
        )

        cint = sequence["cint"]
        cint_len = context["cint_len"]

        return {"char": cint, "char_len": cint_len}

    def create_iterator(self, data_files):
        """Create iterator for data."""
        data_set = tf.data.TFRecordDataset(data_files)
        data_set = data_set.map(self.get_instance)
        data_set = data_set.shuffle(buffer_size=10000)
        data_set = data_set.padded_batch(
            self.batch_size, padded_shapes={'char': [None], 'char_len':[]})

        data_iter = data_set.make_initializable_iterator()
        return data_set, data_iter
