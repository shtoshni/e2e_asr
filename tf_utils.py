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

def get_variables_available_in_checkpoint(variables, checkpoint_path):
    """Returns the subset of variables available in the checkpoint.

    Inspects given checkpoint and returns the subset of variables that are
    available in it.

    TODO: force input and output to be a dictionary.

    Args:
      variables: a list or dictionary of variables to find in checkpoint.
      checkpoint_path: path to the checkpoint to restore variables from.

    Returns:
      A list or dictionary of variables.
    Raises:
      ValueError: if `variables` is not a list or dict.
    """
    try:
        if isinstance(variables, list):
            variable_names_map = {variable.op.name: variable for variable in variables}
        elif isinstance(variables, dict):
            variable_names_map = variables
        else:
            raise ValueError('`variables` is expected to be a list or dict.')
        ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
        ckpt_vars = ckpt_reader.get_variable_to_shape_map().keys()
        vars_in_ckpt = {}
        for variable_name, variable in sorted(variable_names_map.items()):
            if variable_name in ckpt_vars:
                print ("Found: %s" %variable_name)
                vars_in_ckpt[variable] = ckpt_reader.get_tensor(variable_name)
        return vars_in_ckpt
    except:
        return {}


def restore_common_variables(sess, ckpt_path):
    """Restore common variables from checkpoint."""
    common_vars = get_variables_available_in_checkpoint(
        tf.trainable_variables(), ckpt_path)
    for var in common_vars:
        try:
            sess.run(var.assign(common_vars[var]))
            print ("Using pre-trained: %s" %var.name)
        except ValueError:
            print ("Shape wanted: %s, Shape stored: %s for %s"
                   %(str(var.shape), str(common_vars[var].shape), var.name))


def get_matching_variables(var_name_substr, checkpoint_path):
    """Returns the subset of variables available in the checkpoint.

    Inspects given checkpoint and returns the subset of variables that are
    available in it.

    TODO: force input and output to be a dictionary.

    Args:
      variables: a list or dictionary of variables to find in checkpoint.
      checkpoint_path: path to the checkpoint to restore variables from.

    Returns:
      A list or dictionary of variables.
    Raises:
      ValueError: if `variables` is not a list or dict.
    """
    ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
    ckpt_vars = ckpt_reader.get_variable_to_shape_map().keys()
    vars_in_ckpt = {}
    for var_name in ckpt_vars:
        if var_name_substr in var_name:
            if "Adam" not in var_name:
                vars_in_ckpt[var_name] = ckpt_reader.get_tensor(var_name)
    return vars_in_ckpt
