"""Class for working with speech datasets."""

import glob
import tensorflow as tf

class SpeechDataset(object):
    """Dataset class for speech datasets."""

    def __init__(self, params, pattern, isTraining):
        self.params = params  # batch_size, max_output, feat_length
        self.is_training = isTraining
        self.create_iterator(pattern)

    def get_files(self, pattern):
        """Given a pattern return all files matching pattern in data_dir."""
        all_files = glob.glob(self.params.data_dir + "/" + pattern + "*")
        #all_files = glob.glob(FLAGS.data_dir + "/train*1.1.*")
        all_files.sort()
        print ("Number of %s files: %d" %(pattern, len(all_files)))
        return all_files

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

    def create_iterator(self, pattern):
        """Create iterator for data."""
        data_files = self.get_files(pattern)
        data_set = tf.data.TFRecordDataset(data_files)
        data_set = data_set.map(self.get_instance)
        if self.is_training:
            data_set = data_set.shuffle(buffer_size=10000)
        data_set = data_set.padded_batch(
            self.params.batch_size, padded_shapes={'logmel': [None, self.params.feat_length],
                                                   'char': [None], 'phone': [None],
                                                   'logmel_len':[], 'char_len':[], 'phone_len':[]})

        self.data_iter = data_set.make_initializable_iterator()

    def initialize_iterator(self):
        """Return data iterator."""
        return self.data_iter.initializer

    def get_batch(self):
        """Get a batch from the iterator."""
        batch = self.data_iter.get_next()
        encoder_inputs = batch["logmel"]
        encoder_len = batch["logmel_len"]

        decoder_inputs = {}
        decoder_len = {}
        for task in ["char", "phone"]:
            decoder_inputs[task] = tf.transpose(batch[task], [1, 0])
            decoder_len[task] = batch[task + "_len"]
            if not self.is_training:
                decoder_len[task] = tf.ones_like(decoder_len[task]) *\
                    self.params.max_output

        return [encoder_inputs, decoder_inputs, encoder_len, decoder_len]
