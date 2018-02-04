from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
from six.moves import zip

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
import tensorflow.contrib.rnn as rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear
from tensorflow.python.ops.rnn_cell_impl import _linear as linear
from tensorflow.contrib.cudnn_rnn import CudnnLSTMSaveable as CudnnLSTM
import tensorflow as tf


def _extract_argmax_and_embed(embedding, output_projection=None):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(
          prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    return emb_prev
  return loop_function


def _sample_and_embed(embedding):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """
  def loop_function(prev):
    prev_symbol = tf.reshape(tf.multinomial(prev, 1), [-1])
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    return emb_prev
  return loop_function


def attention_decoder(decoder_inputs, attention_states, cell, seq_len_inp, seq_len, 
                    use_conv=False, conv_filter_width=10, conv_num_channels=10,
                    output_size=None, num_heads=1, loop_function=None,
                    dtype=dtypes.float32, scope=None, isTraining=True,
                    initial_state_attention=False, attention_vec_size=None):
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if output_size is None:
    output_size = cell.output_size

  with variable_scope.variable_scope(scope or "attention_decoder", \
          initializer=tf.random_uniform_initializer(-0.075, 0.075)):
    batch_size = array_ops.shape(decoder_inputs)[1]  # Needed for reshaping.
    attn_length = tf.shape(attention_states)[1]
    attn_size = attention_states.get_shape()[2].value

    emb_size = decoder_inputs.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = tf.expand_dims(attention_states, 2)
    hidden_features = []
    v = []
    if use_conv:
        F = []
        U = []

    attention_vec_size = 128#attn_size  # Size of query vectors for attention.

    for a in xrange(num_heads):
      k = variable_scope.get_variable("AttnW_%d" % a,
                                      [1, 1, attn_size, attention_vec_size])
      hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))
      if use_conv:
        F.append(variable_scope.get_variable("AttnF_%d" % a, 
                                    [conv_filter_width, 1, 1, conv_num_channels]))   
        U.append(variable_scope.get_variable("AttnU_%d" % a, 
                                    [1, 1, conv_num_channels, attention_vec_size]))
    
        
    batch_attn_size = array_ops.stack([batch_size, attn_size])
    attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
         for _ in xrange(num_heads)]
    for a in attns:  # Ensure the second shape of attention vectors is set.
        a.set_shape([None, attn_size])

    batch_alpha_size = array_ops.stack([batch_size, attn_length, 1, 1])
    alphas = [array_ops.zeros(batch_alpha_size, dtype=dtype)
             for _ in xrange(num_heads)]
    #for a in alphas:  
    #    a.set_shape([None, attn_length, 1, 1])

    ## Assumes Time major arrangement
    inputs_ta = tf.TensorArray(size=400, dtype=tf.float32)
    inputs_ta = inputs_ta.unstack(decoder_inputs)
    
    attn_mask = tf.sequence_mask(tf.cast(seq_len_inp, tf.int32), dtype=tf.float32)

    def raw_loop_function(time, cell_output, state, loop_state):
        def attention(query, prev_alpha):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            alphas = []
            for a in xrange(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    attn_proj = _Linear(query, attention_vec_size, True)
                    y = attn_proj(query)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    if use_conv:
                        conv_features = nn_ops.conv2d(prev_alpha[a], F[a], [1, 1, 1, 1], "SAME")
                        feat_reshape = nn_ops.conv2d(conv_features, U[a], [1, 1, 1, 1], "SAME")

                        # Attention mask is a softmax of v^T * tanh(...).
                        s = math_ops.reduce_sum(
                            v[a] * math_ops.tanh(hidden_features[a] + y + feat_reshape), [2, 3])
                    else:
                        s = math_ops.reduce_sum(
                            v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])

                    #alpha = nn_ops.softmax(s)
                    alpha = nn_ops.softmax(s) * attn_mask
                    sum_vec = tf.reduce_sum(alpha, reduction_indices=[1], keep_dims=True)
                    norm_term = tf.tile(sum_vec, tf.stack([1, tf.shape(alpha)[1]]))
                    alpha = alpha / norm_term

                    alpha = tf.expand_dims(alpha, 2)
                    alpha = tf.expand_dims(alpha, 3)#array_ops.reshape(alpha, [-1, attn_length, 1, 1])
                    alphas.append(alpha)
                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(alpha * hidden, [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return tuple([tuple(ds), tuple(alphas)])

        # If loop_function is set, we use it instead of decoder_inputs.
        elements_finished = (time >= seq_len)
        finished = tf.reduce_all(elements_finished)


        if cell_output is None:
            next_state = cell.zero_state(batch_size, dtype=tf.float32)#initial_state
            output = None
            loop_state = tuple([tuple(attns), tuple(alphas)])
            next_input = inputs_ta.read(time)
        else:
            next_state = state
            loop_state = attention(cell_output, loop_state[1])
            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + list(loop_state[0]), output_size, True)

            if not isTraining:
                simple_input = loop_function(output, time) 
                print ("Evaluation")
                #capt_time = tf.Print(time)
            else:
                if loop_function is not None:
                    print("Sampling")
                    random_prob = tf.random_uniform([])
                    simple_input = tf.cond(finished,
                        lambda: tf.zeros([batch_size, emb_size], dtype=tf.float32),
                        lambda: tf.cond(tf.less(random_prob, 0.9), 
                            lambda: inputs_ta.read(time), 
                            lambda: loop_function(output))
                        )
                else:
                    print ("No Sampling")
                    simple_input = tf.cond(finished,
                        lambda: tf.zeros([batch_size, emb_size], dtype=tf.float32),
                        lambda: inputs_ta.read(time)
                        )

            # Merge input and previous attentions into one vector of the right size.
            input_size = simple_input.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input")
            with variable_scope.variable_scope("InputProjection"):
                next_input_proj = _Linear([simple_input] + list(loop_state[0]), input_size, True)
                next_input = next_input_proj([simple_input] + list(loop_state[0]))

        return (elements_finished, next_input, next_state, output, loop_state)

  outputs, state, _ = rnn.raw_rnn(cell, raw_loop_function)
  return outputs.concat(), state


def embedding_attention_decoder(decoder_inputs, attention_states,
                                cell, seq_len_inp, seq_len_target, num_symbols, embedding_size, 
                                use_conv=False, conv_filter_width=5, 
                                conv_num_channels=20, 
                                num_heads=1,
                                output_size=None, output_projection=None,
                                feed_previous=False, reuse=False, 
                                update_embedding_for_previous=False,
                                dtype=dtypes.float32, scope=None,
                                initial_state_attention=False,
                                sch_samp=True):
  if output_size is None:
    output_size = cell.output_size
  if output_projection is not None:
    proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with variable_scope.variable_scope(scope or "embedding_attention_decoder", 
          initializer=tf.random_uniform_initializer(-0.075, 0.075), reuse=reuse):
    embedding = variable_scope.get_variable("embedding",
            [num_symbols, embedding_size], initializer=tf.random_uniform_initializer(-2.0, 2.0))
    if feed_previous:
        loop_function = _extract_argmax_and_embed(
            embedding, output_projection)
    elif "char" in scope and sch_samp:
        loop_function = _sample_and_embed(embedding) 
    else:
        loop_function = None
    emb_inp = embedding_ops.embedding_lookup(embedding, decoder_inputs) 
    return attention_decoder(
        emb_inp, attention_states, cell, seq_len_inp, seq_len_target,  
        use_conv=use_conv, conv_filter_width=conv_filter_width,  ## Convolution feature param
        conv_num_channels=conv_num_channels,                    
        output_size=output_size, isTraining=(not feed_previous),
        num_heads=num_heads, loop_function=loop_function,
        initial_state_attention=initial_state_attention)
        

def embedding_attention_seq2seq(task_to_id, encoder_inputs, decoder_inputs, 
                                seq_len, seq_len_target, feed_previous,   
                                cell, cell_dec, num_layers, 
                                num_decoder_symbols,
                                embedding_size, skip_step=1, bi_dir=False, 
                                use_conv=False, conv_filter_width=5, 
                                conv_num_channels=40, 
                                num_heads=1, output_projection=None,
                                dtype=dtypes.float32,
                                scope=None, initial_state_attention=False,
                                base_pyramid=False, sch_samp=True, 
                                use_dynamic=True):
    
    def encode_input(encoder_inputs, cell, layer_depth, bidir=False, seq_len=None, dtype=dtypes.float32, use_dynamic=True):
        """
        Uses LSTMBlockFusedCell
        """
        with variable_scope.variable_scope("RNNLayer%d" %(layer_depth), \
                initializer = tf.random_uniform_initializer(-0.075, 0.075)):
            seq_len_32 = tf.cast(seq_len, tf.int32)
            if not use_dynamic:
                if bi_dir:
                    encoder_outputs_fw, _ = cell(encoder_inputs, sequence_length=seq_len_32, dtype=dtype)
                    encoder_outputs_bw, _ = cell(tf.reverse_sequence(encoder_inputs, seq_len, seq_axis=0, batch_axis=1), \
                            sequence_length=seq_len_32, dtype=dtype, scope="lstm_bw")
                    encoder_outputs = tf.concat([encoder_outputs_fw, encoder_outputs_bw], 2)
                else:
                    encoder_outputs, _ = cell(encoder_inputs, sequence_length=seq_len_32, dtype=dtype)
                return encoder_outputs
            else:
                if bi_dir:
                    (encoder_output_fw, encoder_output_bw), _ = rnn.bidirectional_dynamic_rnn(cell, cell,
                            encoder_inputs, sequence_length=seq_len, dtype=dtype, time_major=True)
                    encoder_outputs = tf.concat([encoder_output_fw, encoder_output_bw], 2)

                else:
                    encoder_outputs, _ = rnn.dynamic_rnn(cell, 
                            encoder_inputs, sequence_length=seq_len, dtype=dtype, time_major=True)
                return encoder_outputs


    def get_pyramid_input(input_tens, seq_len, skip_step):
        """
        Assumes batch major input tensor - input_tens
        """
        max_seq_len = tf.reduce_max(seq_len)
        check_rem = tf.cast(tf.mod(max_seq_len, skip_step), tf.int32)

        feat_size = input_tens.get_shape()[2].value
        #print ("Feature Size: %d" %feat_size)

        div_input_tens = tf.cond(tf.cast(check_rem, tf.bool), ##Odd or even length
                                    lambda: tf.identity(tf.concat([input_tens, \
                                            tf.zeros([tf.shape(input_tens)[0], skip_step-check_rem, feat_size])], 1)),
                                    lambda: tf.identity(input_tens)
                                    )

        output_tens = tf.reshape(div_input_tens, [tf.shape(div_input_tens)[0], \
                tf.cast(tf.shape(div_input_tens)[1]/skip_step, tf.int32), feat_size * skip_step])
        ## Get the ceil division since we pad it with 0s
        seq_len = tf.to_int64(tf.ceil(tf.truediv(seq_len, tf.cast(skip_step, dtype=tf.int64))))
        return output_tens, seq_len



    with variable_scope.variable_scope(scope or "embedding_attention_seq2seq", \
            initializer = tf.random_uniform_initializer(-0.1, 0.1)):
        attention_states = {}
        time_major_states = {}
        seq_len_inps = {}
        
        for task, num_layer in num_layers.items():
            if task == "char":
                attention_states[num_layer] = None
            else:
                time_major_states[num_layer] = None
            
            seq_len_inps[num_layer] = None
        
        max_depth = max(attention_states.keys())

        resolution_fac = 1  ## Term to maintain time-resolution factor
        if base_pyramid:
            print ("Splicing at base level !!")
            encoder_inputs, seq_len = get_pyramid_input(encoder_inputs, seq_len, skip_step)
            resolution_fac *= 2


        for i in xrange(max_depth):
            layer_depth = i+1
            ## Transpose the input as the LSTMBlockFusedCell requires time major input
            encoder_outputs = encode_input(tf.transpose(encoder_inputs, [1, 0, 2]), cell, \
                    layer_depth, bidir=bi_dir, seq_len=seq_len, dtype=dtype, use_dynamic=use_dynamic) 

            if time_major_states.has_key(layer_depth):
                time_major_states[layer_depth] = encoder_outputs
                print ("Saving time major output for layer %d" %(layer_depth))

            encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])

            if attention_states.has_key(layer_depth):
                attention_states[layer_depth] = encoder_outputs
            
            seq_len_inps[layer_depth] = seq_len

            ## For every character there are rougly 8 frames - we don't want to go beyond that
            if skip_step > 1 and i != (max_depth-1) and resolution_fac <= 8:    
                encoder_inputs, seq_len = get_pyramid_input(encoder_outputs, seq_len, skip_step)
                resolution_fac *= 2
            else:
                encoder_inputs = encoder_outputs
        

        output = {} 
        state = {}
        for task, task_id in task_to_id.iteritems():
            if decoder_inputs.has_key(task):
                output_size = None
                cell = cell_dec[task]
                if output_projection is None:
                    cell = rnn_cell.OutputProjectionWrapper(cell_dec[task], num_decoder_symbols[task])
                    output_size = num_decoder_symbols[task]

                def decoder_f(task, feed_previous, reuse=False):
                    output, _ = embedding_attention_decoder(
                            decoder_inputs[task], 
                            attention_states[num_layers[task]], cell,
                            seq_len_inps[num_layers[task]], seq_len_target[task],
                            num_decoder_symbols[task], embedding_size, 
                            use_conv=use_conv, conv_filter_width=conv_filter_width,
                            conv_num_channels=conv_num_channels,
                            num_heads=num_heads,
                            output_size=output_size, output_projection=output_projection,
                            feed_previous=feed_previous, reuse=reuse,
                            scope="embedding_attention_decoder_" + task, sch_samp=sch_samp)
                    return output
                    
                
                if task == "phone":
                    output[task] = {"output": time_major_states[num_layers[task]], 
                                    ## 3 extra symbols - <pad>, <go>, <eos> and add blank symbol
                                    "nb_classes": num_decoder_symbols[task] - 3 + 1, 
                                    "seq_len": seq_len_inps[num_layers[task]]
                                    }
                else:
                    if isinstance(feed_previous, bool):
                        output[task] = decoder_f(task, feed_previous)
                    else:
                        output[task] = tf.cond(feed_previous,
                                            lambda: decoder_f(task, True),
                                            lambda: decoder_f(task, False, reuse=True))

    return output


def sequence_loss(logits, targets, weights, seq_len,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
    with ops.name_scope(name, "sequence_loss", [logits, targets, weights]):
        flat_targets = tf.reshape(targets, [-1])
        cost = nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=flat_targets)
        mask_cost = weights * cost
        loss = tf.reshape(mask_cost, tf.shape(targets))## T*B
        cost_per_example = tf.reduce_sum(loss, reduction_indices=0) / tf.cast(seq_len, dtypes.float32)##Reduce across time steps
         
        return tf.reduce_mean(cost_per_example)


def ctc_loss(output, targets, seq_len_inp, seq_len_target, num_classes, task):
    out_size = output.get_shape()[2].value
    time_shape = tf.shape(output)[0] ## Time shape

    ## For getting logits, reshape
    output = tf.reshape(output, [-1, out_size])
    with variable_scope.variable_scope("ctc_" + task):
        logits = linear([output], num_classes, True)

    logits = tf.reshape(logits, [time_shape, -1, num_classes])

    ## Keep element indicator batch major
    #elem_ind = tf.transpose(tf.sequence_mask(seq_len), [1, 0])
    elem_ind = tf.sequence_mask(seq_len_target)
    zeros = tf.zeros_like(elem_ind)
    
    ## Element presence cond
    elem_pres_cond = tf.not_equal(elem_ind, zeros)
    
    elem_indices = tf.where(elem_pres_cond)
    ## Batch major target
    batch_major_target = tf.transpose(targets, [1, 0])
    elem_values = tf.cast(tf.gather_nd(batch_major_target, elem_indices), dtypes.int32)
 
    sparse_shape = tf.cast(tf.shape(batch_major_target), dtypes.int64)
    
    target_tensor = tf.SparseTensor(elem_indices, elem_values, sparse_shape)
    
    ctc_loss = tf.nn.ctc_loss(target_tensor, logits, tf.cast(seq_len_inp, dtypes.int32))
    normalized_ctc_loss = ctc_loss/tf.cast(seq_len_target, dtypes.float32)
    return tf.reduce_mean(normalized_ctc_loss)


def model_with_buckets(task_to_id, encoder_inputs, decoder_inputs, targets, weights, 
                        seq_len, seq_len_target, seq2seq, 
                        softmax_loss_function=None, per_example_loss=False, name=None):
    all_inputs = [encoder_inputs]

    for task, task_id in task_to_id.iteritems():
        all_inputs += [decoder_inputs[task], targets[task], weights[task]]
    with ops.name_scope(name, "model_with_buckets", all_inputs):
        with variable_scope.variable_scope(variable_scope.get_variable_scope()):
            outputs = seq2seq(encoder_inputs,
                                    decoder_inputs,
                                    seq_len, seq_len_target) 

            losses = {}
            for task, task_id in task_to_id.iteritems():
                if task == "phone":
                    losses[task] = ctc_loss(outputs[task]["output"], targets[task], 
                            outputs[task]["seq_len"], seq_len_target[task],
                            outputs[task]["nb_classes"], "phone")
                else:
                    losses[task] = sequence_loss(outputs[task], targets[task], 
                        weights[task], seq_len_target[task], softmax_loss_function=softmax_loss_function)

    return outputs, losses
