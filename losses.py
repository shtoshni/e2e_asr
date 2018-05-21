import tensorflow as tf


class LossUtils(object):

    @staticmethod
    def sparse_cross_entropy_loss(logits, targets, seq_len_target):
        """Calculate the cross entropy loss w.r.t. given target.

        Args:
            logits: A 2-d tensor of shape (TxB)x|V| containing the logit score
                per output symbol.
            targets: 2-d tensor of shape TxB that contains the ground truth
                output symbols.
            seq_len_target: Sequence length of output sequences. Required to
                mask padding symbols in output sequences.
        """
        with tf.name_scope("sequence_loss", [logits, targets]):
            flat_targets = tf.reshape(targets, [-1])
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=flat_targets)

            # Mask this cost since the output sequence is padded
            batch_major_mask = tf.sequence_mask(seq_len_target,
                                                dtype=tf.float32)
            time_major_mask = tf.transpose(batch_major_mask, [1, 0])
            weights = tf.reshape(time_major_mask, [-1])
            mask_cost = weights * cost

            loss = tf.reshape(mask_cost, tf.shape(targets))
            # Average the loss for each example by the # of timesteps
            cost_per_example = tf.reduce_sum(loss, reduction_indices=0) /\
                tf.cast(seq_len_target, tf.float32)
            # Return the average cost over all examples
            return tf.reduce_mean(cost_per_example)

    @staticmethod
    def smooth_cross_entropy_loss(logits, targets, num_classes, seq_len_target,
                                  label_smoothing=0.05):
        """Calculate the cross entropy loss w.r.t. given target.

        Args:
            logits: A 2-d tensor of shape (TxB)x|V| containing the logit score
                per output symbol.
            targets: 2-d tensor of shape TxB that contains the ground truth
                output symbols.
            seq_len_target: Sequence length of output sequences. Required to
                mask padding symbols in output sequences.
        """
        with tf.name_scope("sequence_loss", [logits, targets]):
            flat_targets = tf.reshape(targets, [-1])

            onehot_labels = tf.one_hot(flat_targets, num_classes)
            #label_smoothing=0.0)
            num_classes = tf.constant(num_classes, logits.dtype)
            smooth_positives = 1.0 - label_smoothing
            smooth_negatives = label_smoothing / num_classes
            smoothed_labels = onehot_labels * smooth_positives + smooth_negatives

            cost = tf.nn.softmax_cross_entropy_with_logits(
                labels=smoothed_labels, logits=logits)#,

            # Mask this cost since the output sequence is padded
            batch_major_mask = tf.sequence_mask(seq_len_target,
                                                dtype=tf.float32)
            time_major_mask = tf.transpose(batch_major_mask, [1, 0])
            weights = tf.reshape(time_major_mask, [-1])
            mask_cost = weights * cost

            loss = tf.reshape(mask_cost, tf.shape(targets))
            # Average the loss for each example by the # of timesteps
            cost_per_example = tf.reduce_sum(loss, reduction_indices=0) /\
                tf.cast(seq_len_target, tf.float32)
            # Return the average cost over all examples
            return tf.reduce_mean(cost_per_example)
