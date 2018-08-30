import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

import normal_ops


slim = tf.contrib.slim

class MultiGPU():
    def __init__(self, loss_fns,
                 num_loss, num_intermediate,
                 input_tensors, opt_fn=None, use_gpu=None,
                 update_collection_names=None):
        # loss_fns is a function which returns ([losses], [var_lists], [secondary_tensors])
        # loss function takes in **kwargs dict arguments
        # opt_fn: function which takes a variable that tracks global_step
        # use_gpu: list of gpus to use, default to [0]
        # update_collection_names: List of update ops which need to be applied before gradients
        
        if not opt_fn:
            def opt_fn(global_step):
                return tf.train.AdamOptimizer(1e-4)
        self.opt_fn = opt_fn
        
        if type(use_gpu) is int:
            use_gpu = [use_gpu]
        
        self.update_collection_names = update_collection_names or []
        self.use_gpu = use_gpu or [0]
        self.num_loss = num_loss
        self.num_intermediate = num_intermediate
        
        self.loss_fns = loss_fns
        
        input_tensors = {k: tf.split(v, max(len(self.use_gpu), 1)) for k, v in input_tensors.iteritems()}
        self.input_tensors = [{k: v[i] for k, v in input_tensors.iteritems()}
                              for i in range(max(len(self.use_gpu), 1))]
        
        self.total_losses = [0 for i in range(self.num_loss)]
        self.var_list = []
        self.per_grads = [[] for i in range(self.num_loss)]
        
        self.global_steps = [tf.Variable(0, trainable=False) for i in range(self.num_loss)]
        self.optimizers = [self.opt_fn(gs) for gs in self.global_steps]
        self.ts = []
        
        self.intermediate_tensors = [[] for i in range(self.num_intermediate)]
        
        with tf.variable_scope(tf.get_variable_scope()):
            for i, gpu_id in enumerate(self.use_gpu):
                print('Initializing graph on gpu %i' % gpu_id)
                with tf.device('/gpu:%d' % gpu_id):
                    args = self.input_tensors[i]
                    j_losses, var_lists, intermediate_tens = self.loss_fns(**args)
                    [self.intermediate_tensors[k].append(inter) for k, inter in enumerate(intermediate_tens)]
                        
                    
                    for j, j_loss in enumerate(j_losses):
                        self.total_losses[j] += j_loss
                        j_grad = self.optimizers[j].compute_gradients(j_loss,
                                                                      var_list=var_lists[j])
                        self.per_grads[j].append(j_grad)
                tf.get_variable_scope().reuse_variables()
            # may be buggy behavior, batch norm typically aggregates per gpu statistic
            # for image translation ignore since batch norm is left on training normally
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_collections = [tf.get_collection(u) for u in self.update_collection_names]
        elements = []
        [elements.extend(u) for u in update_collections]
        
        for j in range(self.num_loss):
            opt_grad = normal_ops.average_gradients(self.per_grads[j])
            with tf.control_dependencies(update_ops + elements):
                self.ts.append(self.optimizers[j].apply_gradients(opt_grad,
                                                                  global_step=self.global_steps[j]))
                
        with tf.control_dependencies(self.ts):
            self.all_ts = tf.no_op(name='optimizers')
            
        self.intermediate_tensors = [tf.concat(ts, axis=0) for ts in self.intermediate_tensors]
        
        num_gpu = max(1., len(self.use_gpu) + 0.)
        self.total_losses = [tloss / num_gpu for tloss in self.total_losses]
    
    def get_ts(self):
        return self.all_ts

    def get_losses(self):
        return self.total_losses
    
    def get_intermediate(self):
        return self.intermediate_tensors
        
            
            
                        
        
        