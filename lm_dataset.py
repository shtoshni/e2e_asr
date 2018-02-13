"""Class for working with speech datasets."""

import glob
import tensorflow as tf

class LMDataset(object):
    """Dataset class for language model dataset."""

    def __init__(self, filenames, batch_size):
        self.batch_size = batch_size
        self.create_iterator(filenames)

    def get_instance(self, proto):
        """Parse the proto to prepare instance."""
        context_features = {
            "segment": tf.FixedLenFeature([], tf.string),
            "logmel_len": tf.FixedLenFeature([], tf.int64),
            "cint_len": tf.FixedLenFeature([], tf.int64),
            "pint_len": tf.FixedLenFeature([], tf.int64),
        }
        sequence_features = {
            "logmel": tf.FixedLenSequenceFeature(shape=[self.params.feat_length], dtype=tf.float32),
            "cint": tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64),
            "pint": tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64)
        }
        # parse a sequence example given the above instructions on the structure
        context, sequence = tf.parse_single_sequence_example(
            serialized=proto,
            context_features=context_features,
            sequence_features=sequence_features
        )
        # unpack segment ID
        # segmentID = context["segment"]

        logmel = sequence["logmel"]
        cint = sequence["cint"]
        pint = sequence["pint"]

        logmel_len = context["logmel_len"]
        cint_len = context["cint_len"]
        pint_len = context["pint_len"]

        return {"logmel": logmel, "char": cint, "phone":pint,
                "logmel_len": logmel_len, "char_len": cint_len,
                "phone_len": pint_len}

    def create_iterator(self, data_files):
        """Create iterator for data."""
        data_set = tf.data.TFRecordDataset(data_files)
        data_set = data_set.map(self.get_instance)
        data_set = data_set.shuffle(buffer_size=10000)
        data_set = data_set.padded_batch(
            self.batch_size, padded_shapes={'char': [None], 'char_len':[]})

        self.data_iter = data_set.make_initializable_iterator()

    def initialize_iterator(self):
        """Return data iterator."""
        return self.data_iter.initializer

    def get_batch(self):
        """Get a batch from the iterator."""
        batch = self.data_iter.get_next()
        encoder_inputs = tf.transpose(batch["char"], [1, 0])
        encoder_len = batch["char_len"]

        return [encoder_inputs, encoder_len]
