import tensorflow as tf


def create_shifted_targets(dec_input, seq_len):
    """Shift the dec_input by 1 to create the targets.
    Also creates the mask for loss of dim (T X B)"""
    targets = tf.slice(dec_input, [1, 0], [-1, -1])
    batch_major_mask = tf.sequence_mask(seq_len, dtype=tf.float32)  # B*T
    time_major_mask = tf.transpose(batch_major_mask, [1, 0])  # T*B
    target_weights = tf.reshape(time_major_mask, [-1])

    return targets, target_weights

def get_summary(value, tag):
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
