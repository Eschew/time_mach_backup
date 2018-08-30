import tensorflow as tf
import warnings


NO_OPS = 'NO_OPS'
SPECTRAL_NORM_UPDATE_OPS = "spectral_norm_update_ops"

def scope_has_variables(scope):
  return len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)) > 0

def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def sn_deconv(inp, output_shape, kernel_size=4, strides=2,
              name="deconv2d", kernel_initializer=None,
              update_collection=SPECTRAL_NORM_UPDATE_OPS):
  with tf.variable_scope(name) as scope:
    if scope_has_variables(scope):
      scope.reuse_variables()
    w = tf.get_variable("W", [kernel_size, kernel_size,
                              output_shape[-1],
                              inp.get_shape().as_list()[-1]],
                        initializer=kernel_initializer)
    w_bar = spectral_normed_weight(w, update_collection=update_collection)
    deconv = tf.nn.conv2d_transpose(inp, w_bar,
                                    output_shape=output_shape,
                                    strides=[1, strides, strides, 1],
                                    padding="SAME")
    biases = tf.get_variable("b", [output_shape[-1]],
                             initializer=tf.constant_initializer(0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), output_shape)
    return deconv
                        

def spectral_normed_weight(W, u=None, num_iters=1, update_collection=None, with_sigma=False):
  # Usually num_iters = 1 will be enough
  W_shape = W.shape.as_list()
  W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
  if u is None:
    u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
  def power_iteration(i, u_i, v_i):
    v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
    u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
    return i + 1, u_ip1, v_ip1
  _, u_final, v_final = tf.while_loop(
    cond=lambda i, _1, _2: i < num_iters,
    body=power_iteration,
    loop_vars=(tf.constant(0, dtype=tf.int32),
               u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
  )
  if update_collection is None:
    warnings.warn('Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
                  '. Please consider using a update collection instead.')
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    with tf.control_dependencies([u.assign(u_final)]):
      W_bar = tf.reshape(W_bar, W_shape)
  else:
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    W_bar = tf.reshape(W_bar, W_shape)
    # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
    # has already been collected on the first call.
    if update_collection != NO_OPS:
      tf.add_to_collection(update_collection, u.assign(u_final))
  if with_sigma:
    return W_bar, sigma
  else:
    return W_bar