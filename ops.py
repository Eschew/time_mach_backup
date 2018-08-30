import tensorflow as tf

import numpy as np
slim = tf.contrib.slim


def deconv(batch_input, out_channels, output_shape):
    _b, h, w, _c = batch_input.shape
    resized_input = tf.image.resize_images(batch_input,
                                           output_shape, 
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return slim.conv2d(resized_input, out_channels,
                       kernel_size=4, stride=1, activation_fn=None)

def bn_conv2d(batch_input, out_channels, is_training, kernel_size=4, stride=1):
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #   train_op = optimizer.minimize(loss)
    return slim.batch_norm(conv2d(batch_input, out_channels, kernel_size, stride),
                           is_training=is_training)

def conv2d(batch_input, out_channels, kernel_size=4, stride=1):
    return slim.conv2d(batch_input, out_channels,
                       kernel_size=kernel_size, stride=stride, activation_fn=None)

def bn_deconv(batch_input, out_channels, output_shape, is_training):
    return slim.batch_norm(deconv(batch_input, out_channels, output_shape),
                           is_training=is_training)


def conv_residual_block(batch_inp, units, depth_out, stride=1, is_training=True):
    # takes in non-activated tensors
    depth_in = tf.shape(batch_inp)[-1]
    
    if stride==1 and depth_out == depth_in:
        l0 = batch_inp
    else:
        l0 = bn_conv2d(tf.nn.relu(batch_inp), depth_out, kernel_size=1, stride=stride,
                       is_training=is_training)
    
    residual = bn_conv2d(tf.nn.relu(batch_inp), depth_out, stride=stride,
                         is_training=is_training)
    for i in range(units):
        residual = bn_conv2d(tf.nn.relu(residual), depth_out, stride=1,
                             is_training=is_training)
        
    # non-bn conv
    residual = conv2d(residual, depth_out, stride=1)
    return residual + l0

def deconv_residual_block(batch_inp, units, depth_out,
                          output_shape, is_training=True):
    depth_in = tf.shape(batch_inp)[-1]
    
    l0 = bn_deconv(tf.nn.relu(batch_inp), depth_out, output_shape,
                   is_training=is_training)
    
    for i in range(units):
        residual = bn_conv2d(tf.nn.relu(l0), depth_out, stride=1,
                             is_training=is_training)
    
    # non-bn conv
    residual = conv2d(residual, depth_out, stride=1)
    return residual + l0
    
    
    
    