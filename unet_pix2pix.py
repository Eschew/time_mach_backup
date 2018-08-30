import tensorflow as tf

import numpy as np
slim = tf.contrib.slim



def deconv(batch_input, out_channels, output_shape):
    _b, h, w, _c = batch_input.shape
    resized_input = tf.image.resize_images(batch_input,
                                           output_shape, 
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return slim.batch_norm(slim.conv2d(resized_input, out_channels,
                           kernel_size=4, stride=1, activation_fn=None))

def bn_conv2d(batch_input, out_channels, stride=1):
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #   train_op = optimizer.minimize(loss)
    return slim.batch_norm(slim.conv2d(batch_input, out_channels,
                           kernel_size=4, stride=stride, activation_fn=None))

def bn_deconv(batch_input, out_channels, output_shape):
    return slim.batch_norm(deconv(batch_input, out_channels, output_shape))
    

class P2P():
    
    def __init__(self, ngf=16, ndf=16, lr=1e-4):
        self.ngf = ngf
        self.ndf = ndf
        self.lr = lr
        self.opt = tf.train.AdamOptimizer(self.lr)
        
    def model(self, inp, target, reuse=False):
        with tf.variable_scope("unet", reuse=reuse):
            self.gen_output = self.generator(inp, reuse=reuse)
            gen_pair_input = tf.concat([inp, self.gen_output], axis=3)
            real_pair_input = tf.concat([inp, target], axis=3)

            self.g_d_logit = self.discriminator(gen_pair_input, reuse=reuse)
            self.g_d_pred = tf.nn.sigmoid(self.g_d_logit)
            self.r_d_logit = self.discriminator(real_pair_input, reuse=True)
            self.r_d_pred = tf.nn.sigmoid(self.r_d_logit)
            
        with tf.name_scope("generator_loss"):
            self.g_loss = tf.reduce_mean(-tf.log(self.g_d_pred + 1e-12)) + 1000. * tf.reduce_mean(tf.abs(self.gen_output - target))
        
        with tf.name_scope("discrim_loss"):
            self.d_loss = tf.reduce_mean(-tf.log(self.r_d_pred + 1e-12) + tf.log(1 - self.g_d_pred + 1e-12)) / 2.
            
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_step = self.opt.minimize(self.d_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                    "unet/discrim"))
            self.g_step = self.opt.minimize(self.g_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                    "unet/generator"))
        self.ts = [self.d_step, self.g_step]
        self.losses = [self.d_loss, self.g_loss]
        
        
            
    def discriminator(self, inp, reuse=False):
        ndf = self.ndf
        with tf.variable_scope("discrim", reuse=reuse):
            net = bn_conv2d(inp, ndf, stride=2)
            for i in range(1, 4):
                net = tf.nn.relu(net)
                net = bn_conv2d(net, (2 ** i) * ndf, stride=2)
            
            # logits
            gap = tf.reduce_mean(net, axis=[1, 2])
            net = slim.fully_connected(gap, 256)
            net = slim.fully_connected(gap, 128)
            # [pr(fake), pr(real)]
            return slim.fully_connected(net, 1, activation_fn=None)
            
            
    def generator(self, inp, reuse=False):
        encoding = [] # unet
        with tf.variable_scope("generator", reuse=reuse):
            ngf = self.ngf
            net = bn_conv2d(inp, ngf, stride=2)
            encoding.append(net)
            ngf = ngf * 2

            # relu -> conv -> bn -> save

            # encode
            for i in range(4):
                net = tf.nn.relu(net)
                net = bn_conv2d(net, ngf, stride=2)
                if i==3:
                    # skip last append
                    continue
                encoding.append(net)
                ngf = ngf * 2

            ngf = ngf // 2
            # batch x h // 32 x w // 32 x 32 * ngf
            # decode
            for j in range(4):
                if j == 0:
                    # direct
                    pass
                else:
                    above_layer = encoding[-j]
                    net = tf.concat([net, above_layer], axis=-1)
                net = tf.nn.relu(net)
                net = bn_deconv(net, ngf, tf.shape(encoding[-j - 1])[1:3])
                ngf = ngf // 2

            net = deconv(net, 3, tf.shape(inp)[1:3])
            return tf.nn.tanh(net)
        
                
        
        