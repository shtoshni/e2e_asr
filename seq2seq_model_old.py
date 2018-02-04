from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import re
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.rnn import BasicLSTMCell as LSTM
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import LSTMBlockFusedCell
from tensorflow.contrib.cudnn_rnn import CudnnLSTM#Saveable as CudnnLSTM

import data_utils
import many2one_seq2seq

class Seq2SeqModel(object):

  def __init__(self, target_vocab_size, buckets, hidden_size, hidden_size_decoder,
            num_layers, num_layers_decoder, embedding_size, skip_step, bi_dir, use_conv,
            conv_filter_width, conv_num_channels, max_gradient_norm, batch_size, learning_rate,
            learning_rate_decay_factor, optimizer, data_limits, queue=None, use_lstm=False,
            output_keep_prob=0.8, num_samples=512, forward_only=False, task_to_id=None,
            base_pyramid=False, sch_samp=True, apply_dropout=False, l2_weight=0, avg=False):
    self.target_vocab_size = target_vocab_size
    self.batch_size = batch_size
    self.buckets = buckets
    self.task_to_id = task_to_id
    self.data_limits = data_limits

    self.queue = queue

    self.epoch = tf.Variable(0, trainable=False)
    self.epoch_incr = self.epoch.assign(self.epoch + 1)

    self.bi_dir = bi_dir

    self.learning_rate = tf.get_variable("lr", shape=[], dtype=tf.float32, \
            initializer=tf.constant_initializer(learning_rate), trainable=False)
    #self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(
        tf.maximum(self.learning_rate * learning_rate_decay_factor, \
                    tf.constant(1e-4, dtype=tf.float32)))
    self.global_step = tf.Variable(0, trainable=False)

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None

    if forward_only:
        ## Ensure that during evaluation scheduled sampling is not used
        sch_samp = False

    if apply_dropout:
        #cell = LSTM(hidden_size, state_is_tuple=True)
    else:
        cell = LSTMBlockFusedCell(hidden_size)
    decoder_cell = LSTM(hidden_size_decoder, state_is_tuple=True)
    #decoder_cell = LSTMBlockCell(hidden_size_decoder)
    if (not forward_only) and apply_dropout:
        ## Always use the wrapper - To not use dropout just make the probability 1
        print("Dropout used !!")
        #cell = DropoutWrapper(cell, output_keep_prob=output_keep_prob)
        decoder_cell = DropoutWrapper(decoder_cell, output_keep_prob=output_keep_prob)

    cell_dec = {}
    for task, task_id in task_to_id.iteritems():
        if num_layers_decoder[task] == 1:
            ## Not same depth as encoder
            task_cell_dec = decoder_cell
        else:
            task_cell_dec = MultiRNNCell([decoder_cell] * \
                    num_layers_decoder[task], state_is_tuple=True)
            print ("MultiRNN for %s decoder" %(task))
        cell_dec[task] = task_cell_dec


    def seq2seq_f(encoder_inputs, decoder_inputs, seq_len, seq_len_target, do_decode):
        return many2one_seq2seq.embedding_attention_seq2seq(
            task_to_id, encoder_inputs, decoder_inputs,
            seq_len, seq_len_target, do_decode,
            cell, cell_dec, num_layers,
            target_vocab_size,
            embedding_size,
            skip_step=skip_step,
            bi_dir=bi_dir,
            use_conv=use_conv, conv_filter_width=conv_filter_width,
            conv_num_channels=conv_num_channels,
            output_projection=output_projection,
            base_pyramid=base_pyramid,
            sch_samp=sch_samp,
            use_dynamic=apply_dropout)


    if forward_only:
        ## Batch major
        self.encoder_inputs = tf.placeholder(tf.float32, \
                shape=[None, None, self.data_limits.FEAT_LEN], name='encoder')

        _batch_size = tf.shape(self.encoder_inputs)[0]
        self.seq_len = tf.fill(tf.expand_dims(_batch_size, 0), tf.constant(2, dtype=tf.int64))
        self.seq_len_target = {}
        for task in task_to_id:
            self.seq_len_target[task] = tf.fill(tf.expand_dims(_batch_size, 0), tf.constant(2, dtype=tf.int32))


        self.decoder_inputs = {}
        self.targets = {}
        self.target_weights = {}
        for task, task_id in task_to_id.iteritems():
            ## (T+1)*B -> EOS is an extra symbol as input but seq_len avoids that
            self.decoder_inputs[task] = tf.placeholder(tf.int32, shape=[None, None], name="decoder_" + task)
            ## Targets are shifted by one - T*B
            self.targets[task] = tf.slice(self.decoder_inputs[task], [1, 0], [-1, -1])

            batch_major_mask = tf.sequence_mask(self.seq_len_target[task], dtype=tf.float32) ## B*T
            time_major_mask = tf.transpose(batch_major_mask, [1, 0]) ## T*B
            self.target_weights[task] = tf.reshape(time_major_mask, [-1])

        self.feed_forward = tf.placeholder(tf.bool, name="feed_forward")

        self.outputs, self.losses = many2one_seq2seq.model_with_buckets(
          task_to_id, self.encoder_inputs, self.decoder_inputs, self.targets,
          self.target_weights, self.seq_len, self.seq_len_target,
          lambda w, x, y, z : seq2seq_f(w, x, y, z, self.feed_forward),
          softmax_loss_function=softmax_loss_function)


    else:
        # Add learning rate summary
        tf.summary.scalar('lr (in 1e3)', self.learning_rate * 1e3)

        # Training outputs and losses.
        self.encoder_inputs, self.decoder_inputs, self.seq_len, self.seq_len_target = self.get_queue_batch()
        self.targets = {}
        self.target_weights = {}
        for task, task_id in task_to_id.iteritems():
            self.seq_len_target[task] = tf.cast(self.seq_len_target[task], tf.int32)
        for task, task_id in task_to_id.iteritems():
            if task == "phone":
                self.targets[task] = self.decoder_inputs[task]
                self.target_weights[task] = None
            else:
                self.targets[task] = tf.slice(self.decoder_inputs[task], [1, 0], [-1, -1])
                batch_major_mask = tf.sequence_mask(self.seq_len_target[task], dtype=tf.float32) ## B*T
                time_major_mask = tf.transpose(batch_major_mask, [1, 0]) ## T*B
                self.target_weights[task] = tf.reshape(time_major_mask, [-1])

        self.outputs, self.losses = many2one_seq2seq.model_with_buckets(
          task_to_id, self.encoder_inputs, self.decoder_inputs, self.targets,
          self.target_weights, self.seq_len, self.seq_len_target,
          lambda w, x, y, z: seq2seq_f(w, x, y, z, False),
          softmax_loss_function=softmax_loss_function)

        for task in task_to_id:
            tf.summary.scalar('Negative log likelihood ' + task, self.losses[task])
        opt = tf.train.AdamOptimizer(self.learning_rate)
        #for var in tf.trainable_variables():
            #if ("weights" in var.name) and ("lstm" in var.name):
                ## Save frobenius norm
                #tf.summary.scalar('Frobenius Norm - ' + var.name, tf.norm(var, ord='euclidean'))

        l2_loss = 0
        if l2_weight > 0:
            for var in tf.trainable_variables():
                if ("biases" in var.name) or ("lstm" in var.name):
                    ## Don't apply L2 on bias or LSTM weight matrix
                    continue
                else:
                    l2_loss += tf.nn.l2_loss(var)

        loss = l2_loss * l2_weight ## Initialize with l2-loss
        for task in task_to_id:
            loss += self.losses[task]

        if avg:
            print ("Averaging loss !!")
            loss /= float(len(task_to_id))

        for variable in tf.trainable_variables():
            print ("%s - %s" %(variable.name, str(variable.get_shape())))
        gradients = tf.gradients(loss, tf.trainable_variables(),
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        self.gradient_norms = norm
        tf.summary.scalar('Gradient Norm', self.gradient_norms)

        self.updates = opt.apply_gradients(\
            zip(clipped_gradients, tf.trainable_variables()), global_step=self.global_step)


    act_output = tf.cast(tf.argmax(self.outputs['char'], axis=1), tf.int64)
    target_resh = tf.cast(tf.reshape(self.targets['char'], [-1]), tf.int64)
    output_diff = tf.cast(tf.not_equal(act_output, target_resh), tf.float32)
    masked_diff = tf.multiply(output_diff, self.target_weights['char'])
    total_errs = tf.reduce_sum(masked_diff)

    self.total_chars = tf.reduce_sum(self.target_weights['char'])
    self.frac_errs = total_errs/tf.cast(self.total_chars, tf.float32)

    #tf.summary.scalar('Char Error', self.frac_errs)

    self.merged = tf.summary.merge_all()
    #self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)


  def step(self, session, encoder_inputs, decoder_inputs, seq_len,
            seq_len_target, isTraining, feed_forward=True):
    # Check if the sizes match.
    task = decoder_inputs.keys()[0]
    task_id = self.task_to_id[task]

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    input_feed[self.encoder_inputs.name] = encoder_inputs
    input_feed[self.decoder_inputs[task].name] = decoder_inputs[task]

    ## SHUBHAM - Feed seq_len as well
    input_feed[self.seq_len.name] = seq_len
    input_feed[self.seq_len_target[task].name] = seq_len_target[task]


    if isTraining:
        ## Regular training
        output_feed = [self.updates,  # Update Op that does SGD.
                     self.gradient_norms,  # Gradient norm.
                     self.losses[task]]  # Loss for this batch.
    else:
        input_feed[self.feed_forward.name] = feed_forward
        if feed_forward:
            ## Regular Evaluation
            output_feed = [self.losses[task]]  # Loss for this batch.
            output_feed.append(self.outputs[task])
        else:
            ## Testing for 0-1 loss
            output_feed = [self.frac_errs]
            output_feed.append(self.losses[task])  # Loss for this batch.
            output_feed.append(self.total_chars)

    outputs = session.run(output_feed, input_feed)
    if isTraining:
        return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
        if feed_forward:
            return None, outputs[0], outputs[1]  # No gradient norm, loss, outputs
        else:
            return outputs


  def get_batch(self, data, bucket_id, task, do_eval=False, sample_eval=False):
    """Get sequential batch"""
    task_id = self.task_to_id[task]
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], {}
    if sample_eval:
        this_batch_size = self.batch_size
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
            seq_len_target[task][i] = decoder_size[task_id]
        else:
            seq_len_target[task][i] = len(decoder_input) + 1
        #print (len(decoder_input) + 1)

    ## Get maximum lengths
    max_len_source = max(seq_len)
    max_len_target = {}
    if do_eval:
        max_len_target[task] = decoder_size[task_id]
    else:
        max_len_target[task] = max(seq_len_target[task])


    for i, sample in enumerate(data_source):
        ## Text input, parse output, speech input
        encoder_input, decoder_input = sample[:2]

        # Encoder inputs are padded and then reversed.
        encoder_pad = [self.data_limits._PAD_VEC] * (max_len_source - seq_len[i])
        ## Pad the speech frames
        #if not self.bi_dir:
        #    encoder_inputs.append(list(reversed(encoder_input)) + encoder_pad)
        #else:
            ## Don't need to pre-reverse
        encoder_inputs.append(encoder_input + encoder_pad)


        ## Don't use seq_len here because the length is artificially set to max for evaluation
        decoder_pad_size = max_len_target[task] - (len(decoder_input) + 1)
        decoder_inputs[task].append([data_utils.GO_ID] + decoder_input +
                [data_utils.EOS_ID] + [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], {}, {}
    #for task, task_id in self.task_to_id:
    batch_decoder_inputs[task] = np.zeros((max_len_target[task] + 1, this_batch_size), dtype=np.int32)

    batch_encoder_inputs = np.array(encoder_inputs)

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(max_len_target[task] + 1):
        for batch_idx in xrange(this_batch_size):
            batch_decoder_inputs[task][length_idx][batch_idx] = decoder_inputs[task][batch_idx][length_idx]

    return batch_encoder_inputs, batch_decoder_inputs, seq_len, seq_len_target


  def get_queue_batch(self):
      """
      IN    bucket_queue        :  queue produced by tf.string_input_producer on list of tfrecords buckets
          nfeats            :  fixed feature dimension for speech input
          batch_kwargs        :  batching parameters (subject to change if batching process changes)
      ----------------------------------------------------------------------------------------------------------------------------------
      OUT    logmels,cints,pints,    :  note this function is SYMBOLIC
          cint_weights,pint_weights,
          logmel_lens,cint_lens,pint_lens,
          segmentIDs
      """
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
        "logmel": tf.FixedLenSequenceFeature(shape=[self.data_limits.FEAT_LEN], dtype=tf.float32),
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
              batch_size=self.batch_size,
              capacity=2000,
              num_threads=1,
              dynamic_pad=True,
              allow_smaller_final_batch=True)
      ## Make cint and pint TIME MAJOR !!!
      #if not self.bi_dir:
          ## No need to reverse if bidirectional
          #logmel = tf.reverse_sequence(logmel, logmel_len, 1)
      cint = tf.transpose(cint, [1, 0])
      pint = tf.transpose(pint, [1, 0])
      return [logmel, {"char": cint, "phone":pint}, logmel_len, {"char": cint_len, "phone": pint_len}]
