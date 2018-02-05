"""Seq2Seq model class that creates the computation graph.

Author: Shubham Toshniwal
Date: February, 2018
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
from bunch import Bunch

import tensorflow as tf

import data_utils
from losses import LossUtils


class Seq2SeqModel(object):
    """Implements the Attention-Enabled Encoder-Decoder model."""

    @classmethod
    def class_params(cls):
        params = Bunch()
        params['batch_size'] = 64
        params['isTraining'] = True
        # Task specification
        params['buckets'] = {'char':[(210, 60), (346, 120), (548, 180), (850, 200),
                                     (1500, 380)],
                             'phone': [(210, 50), (346, 110), (548, 140), (850, 150),
                                       (1500, 250)]}
        params['tasks'] = {'char'}
        params['num_layers'] = {'char':4}
        params['feat_length'] = 80

        # Optimization params
        params['learning_rate'] = 1e-3
        params['learning_rate_decay_factor'] = 0.9
        params['max_gradient_norm'] = 5.0

        # Loss params
        params['avg'] = True

        return params

    def __init__(self, encoder, decoder, tasks, num_layers,
                 queue=None, params=None):
        """Initializer of class that defines the computational graph.

        Args:
            encoder: Encoder object executed via encoder(args)
            decoder: Decoder object executed via decoder(args)
        """
        if params is None:
            self.params = self.class_params()
        else:
            self.params = params

        params = self.params

        self.num_layers = num_layers
        self.queue = queue
        self.tasks = tasks

        self.learning_rate = tf.Variable(float(params.learning_rate),
                                         trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * params.learning_rate_decay_factor)

        # Number of gradient updates performed
        self.global_step = tf.Variable(0, trainable=False)
        # Number of epochs done
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_incr = self.epoch.assign(self.epoch + 1)

        # Batch major
        self.encoder_inputs = tf.placeholder(tf.float32, \
                shape=[None, None, params.feat_length], name='encoder')
        _batch_size = tf.shape(self.encoder_inputs)[0]
        self.seq_len = tf.placeholder(tf.int64, shape=[_batch_size],
                                      name="seq_len")
        # Output sequence length placeholder
        self.seq_len_target = {}
        for task in params.tasks:
            self.seq_len_target[task] = tf.placeholder(
                tf.int64, shape=[_batch_size], name="seq_len_target_" + task)

        self.decoder_inputs = {}
        self.targets = {}
        self.target_weights = {}
        for task in params.tasks:
            # (T+1)*B -> EOS is an extra symbol as input but seq_len_target avoids that
            self.decoder_inputs[task] = tf.placeholder(
                tf.int32, shape=[None, None], name="decoder_" + task)
            # Targets are shifted by one - T*B
            self.targets[task] = tf.slice(self.decoder_inputs[task], [1, 0], [-1, -1])

            batch_major_mask = tf.sequence_mask(self.seq_len_target[task],
                                                dtype=tf.float32)  # B*T
            time_major_mask = tf.transpose(batch_major_mask, [1, 0])  # T*B
            self.target_weights[task] = tf.reshape(time_major_mask, [-1])


        # Create computational graph
        # First encode input
        self.encoder_hidden_states, self.time_major_states, self.seq_len_encs =\
            encoder(self.encoder_inputs, self.seq_len, num_layers)

        self.outputs = {}
        self.losses = {}
        for task in params.tasks:
            task_depth = params.num_layers[task]
            # Then decode
            self.outputs[task] = decoder(
                self.decoder_inputs[task], self.seq_len_target[task],
                self.encoder_hidden_states[task_depth], self.seq_len_encs[task_depth])
            # Training outputs and losses.
            self.losses[task] = LossUtils.seq2seq_loss(
                self.outputs[task], self.targets[task], self.seq_len_target)

            tf.summary.scalar('Negative log likelihood ' + task, self.losses[task])

        if params.isTraining:
            # Gradients and parameter updation for training the model.
            trainable_vars = tf.trainable_variables()
            print ("\nModel parameters:\n")
            for var in trainable_vars:
                print (("{0}: {1}").format(var.name, var.get_shape()))
            print ("\n")
            # Initialize optimizer
            opt = tf.train.AdamOptimizer(self.learning_rate)

            # Add losses across the tasks
            self.total_loss = 0.0
            for task in params.tasks:
                self.total_loss += self.losses[task]

            if params.avg:
                self.total_loss /= float(len(params.tasks))
            tf.summary.scalar('Total loss', self.total_loss)
            # Get gradients from loss
            gradients = tf.gradients(self.total_loss, trainable_vars)
            # Clip the gradients to avoid the problem of gradient explosion
            # possible early in training
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             params.max_gradient_norm)
            self.gradient_norms = norm
            tf.summary.scalar('Gradient Norm', self.gradient_norms)
            # Apply gradients
            self.updates = opt.apply_gradients(
                zip(clipped_gradients, trainable_vars),
                global_step=self.global_step)

        # Model saver function
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)


    def step(self, sess, encoder_inputs, seq_len, decoder_inputs,
             seq_len_target):
        """Perform 1 minibatch update/evaluation.

        Args:
            sess: Tensorflow session where computation graph is created
            encoder_inputs: List of a minibatch of input IDs
            seq_len: Input sequence length
            decoder_inputs: List of a minibatch of output IDs
            seq_len_target: Output sequence length
        Returns:
            Output of a minibatch updated. The exact output depends on
            whether the model is in training mode or evaluation mode.
        """
        # Pass inputs via feed dict method
        params = self.params
        task = decoder_inputs.keys()[0]

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.decoder_inputs[task].name] = decoder_inputs[task]

        # Feed seq_len as well
        input_feed[self.seq_len.name] = seq_len
        input_feed[self.seq_len_target[task].name] = seq_len_target[task]


        if params.isTraining:
            # Regular training
            output_feed = [self.updates,  # Update Op that does SGD.
                           self.gradient_norms,  # Gradient norm.
                           self.losses[task]]  # Loss for this batch.
        else:
            # Testing for 0-1 loss
            output_feed.append(self.losses[task])  # Loss for this batch.
            #output_feed.append(self.total_chars)

        outputs = sess.run(output_feed, input_feed)
        if params.isTraining:
            return outputs[1:]
        else:
            return outputs

    def get_batch(self, data, bucket_id, task, do_eval=False, sample_eval=False):
        """Get sequential batch"""
        params = self.params
        _, decoder_size = params.buckets[task][bucket_id]
        encoder_inputs, decoder_inputs = [], {}
        if sample_eval:
            this_batch_size = params.batch_size
        else:
            this_batch_size = len(data[bucket_id])

        data_source = []
        if sample_eval:
            data_source = random.sample(data[bucket_id], this_batch_size)
        else:
            data_source = data[bucket_id]

        seq_len = np.zeros((this_batch_size), dtype=np.int64)
        seq_len_target = {}
        seq_len_target[task] = np.zeros((this_batch_size), dtype=np.int64)

        decoder_inputs[task] = []

        for i, sample in enumerate(data_source):
            encoder_input, decoder_input = sample[:2]
            seq_len[i] = len(encoder_input)
            if do_eval:
                seq_len_target[task][i] = decoder_size
            else:
                seq_len_target[task][i] = len(decoder_input) + 1

        # Get maximum lengths
        max_len_source = max(seq_len)
        max_len_target = {}
        if do_eval:
            max_len_target[task] = decoder_size
        else:
            max_len_target[task] = max(seq_len_target[task])


        for i, sample in enumerate(data_source):
            # Text input, parse output, speech input
            encoder_input, decoder_input = sample[:2]
            # Encoder inputs are padded and then reversed.
            encoder_pad = [np.zeros(params.feat_length, dtype=np.float32)] * (
                max_len_source - seq_len[i])

            encoder_inputs.append(encoder_input + encoder_pad)


            # Don't use seq_len here because the length is artificially set to max for evaluation
            decoder_pad_size = max_len_target[task] - (len(decoder_input) + 1)
            decoder_inputs[task].append([data_utils.GO_ID] + decoder_input +
                                        [data_utils.EOS_ID] +
                                        [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs = [], {}
        batch_decoder_inputs[task] = np.zeros((max_len_target[task] + 1, this_batch_size),
                                              dtype=np.int32)

        batch_encoder_inputs = np.array(encoder_inputs)

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(max_len_target[task] + 1):
            for batch_idx in xrange(this_batch_size):
                batch_decoder_inputs[task][length_idx][batch_idx] = decoder_inputs[task][batch_idx][length_idx]

        return batch_encoder_inputs, batch_decoder_inputs, seq_len, seq_len_target

    def get_queue_batch(self):
        # -- get reader and read serialized examples from queue
        reader = tf.TFRecordReader()
        _,serialized = reader.read(self.queue)

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
        # ---------------------------------------------------------------------------
        # -- parse a sequence example given the above instructions on the structure
        context,sequence = tf.parse_single_sequence_example(
            serialized=serialized,
            context_features=context_features,
            sequence_features=sequence_features
        )
        # ---------------------------------------------------------------------------
        # -- unpack segment ID
        segmentID = context["segment"]
        # -- ready batch of speech // characters // phonemes
        logmel = sequence["logmel"]
        cint = sequence["cint"]
        pint = sequence["pint"]
        # -- get (non-zero) lengths of sequences
        logmel_len = context["logmel_len"]
        cint_len = context["cint_len"]
        pint_len = context["pint_len"]


        tensor = [
            logmel, cint, pint,            # -- sequences
            logmel_len,cint_len,pint_len,      # -- sequence (nonzero) lengths
            segmentID                # -- segment id, may not be necessary
        ]
        # ---------------------------------------------------------------------------
        # -- output list of tensors each of batch_size x ... dimensions
        logmel, cint, pint, logmel_len, cint_len, pint_len, _ = tf.train.batch(tensors=tensor,
              batch_size=self.params.batch_size,
              capacity=2000,
              num_threads=1,
              dynamic_pad=True,
              allow_smaller_final_batch=True)
        # Make cint and pint TIME MAJOR !!!
        cint = tf.transpose(cint, [1, 0])
        pint = tf.transpose(pint, [1, 0])
        return [logmel, {"char": cint, "phone":pint}, logmel_len, {"char": cint_len, "phone": pint_len}]
