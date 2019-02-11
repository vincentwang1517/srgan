#! /usr/bin/python
# -*- coding: utf8 -*-

import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

# from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.ops import math_ops, init_ops, array_ops, nn
# from tensorflow.python.util import nest
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell

# https://github.com/david-gpu/srez/blob/master/srez_model.py

def initialize_parameters():    
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    return w_init, b_init, g_init

def residual_model1(image, is_train=False, reuse=False, nb_part=3, nb_block=5, first_channel=64):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("residual_model", reuse=reuse):
        n = InputLayer(image, name='input_image')
        nn = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='first_conv')

        for i in range(nb_part):
            with tf.variable_scope("part%d" %i):
                for j in range(nb_block):
                    with tf.variable_scope("block%d" %j):
                        nn = residual_block_dropout(nn, filters=64, is_train=is_train, reuse=reuse)
       
        # nn = ElementwiseLayer([nn, temp], tf.add, name='final_add')
        mean_pool = GlobalMeanPool2d(nn) #[None, 48, 48, 512] -> [None, 512]
        # mean_pool = DropoutLayer(mean_pool, keep=0.7, name='final_dropout')
        logit = DenseLayer(mean_pool, n_units=7, act=None, W_init=w_init, name='final_dense')

        return logit, logit.outputs

def residual_model(image, is_train=False, reuse=False, nb_part=3, nb_block=5, first_channel=64):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("residual_model", reuse=reuse):
        n = InputLayer(image, name='input_image')
        nn = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='first_conv')
        temp = nn
        for i in range(nb_part):
            with tf.variable_scope("part%d" %i):
                for j in range(nb_block):
                    with tf.variable_scope("block%d" %j):
                        nn = residual_block(nn, filters=64, is_train=is_train, reuse=reuse)

        ### ================================ ###
        nn = MaxPool2d(nn, name='maxpool2')
        for i in range(3):
            with tf.variable_scope("part%d" %(i+3)):
                for j in range(3):
                    with tf.variable_scope("block%d" %j):
                        if i==0 and j==0:
                            nn = residual_block(nn, filters=256, is_train=is_train, reuse=reuse, tensor_shape='change')
                        else:
                            nn = residual_block(nn, filters=256, is_train=is_train, reuse=reuse)
        ### ================================= ###
       
        # nn = ElementwiseLayer([nn, temp], tf.add, name='final_add')
        mean_pool = GlobalMeanPool2d(nn) #[None, 48, 48, 512] -> [None, 512]
        logit = DenseLayer(mean_pool, n_units=7, act=None, W_init=w_init, name='final_dense')

        return logit, logit.outputs

def residual_block(input_layer, filters, is_train=False, reuse=False, tensor_shape='change'):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)

    nn = Conv2d(input_layer, filters, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='c1' )
    nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train,gamma_init=g_init, name='b1' )
    nn = Conv2d(nn, filters, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='c2' )
    nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='b2' )

    # If the inputs channel and outputs channel is different
    if tensor_shape == 'change':
        input_layer = Conv2d(input_layer, filters, (1,1), strides=(1,1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='conv_shortcut')
        input_layer = BatchNormLayer(input_layer, act=None, is_train=is_train, gamma_init=g_init, name='bn_shortcut' )

    nn = ElementwiseLayer([input_layer, nn], tf.add, name='residual_add' )
 
    return nn

def residual_block_dropout(input_layer, filters, is_train=False, reuse=False, tensor_shape='change'):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)

    # nn = DropoutLayer(input_layer, keep=0.7, is_train=is_train, name='dropout1')
    nn = Conv2d(input_layer, filters, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='c1' )
    nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train,gamma_init=g_init, name='b1' )
    # nn = DropoutLayer(nn, keep=0.7, is_train=is_train, name='dropout2')
    nn = Conv2d(nn, filters, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='c2' )
    nn = BatchNormLayer(nn, is_train=is_train, act=None, gamma_init=g_init, name='b2' )
    nn = PReluLayer(nn, name='act_prelu')

    # If the inputs channel and outputs channel is different
    if tensor_shape == 'change':
        input_layer = Conv2d(input_layer, filters, (1,1), strides=(1,1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='conv_shortcut')
        input_layer = BatchNormLayer(input_layer, act=None, is_train=is_train, gamma_init=g_init, name='bn_shortcut' )

    nn = ElementwiseLayer([input_layer, nn], tf.add, name='residual_add' )
 
    return nn

def residual_model2(image, is_train=False, reuse=False, nb_part=3, nb_block=5, first_channel=64):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("residual_model", reuse=reuse):
        n = InputLayer(image, name='input_image')
        nn = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='first_conv')

        with tf.variable_scope("part1"):
            for j in range(nb_block):
                with tf.variable_scope("block%d" %j):
                    nn = residual_block_dropout(nn, filters=64, is_train=is_train, reuse=reuse)
            nn = Conv2d(nn, 64, (5,5), (2,2), act=None, padding='SAME',W_init=w_init, b_init=b_init, name='conv_last' )
            nn = BatchNormLayer(nn, act=None, is_train=is_train, gamma_init=g_init, name='bn_last' )
            nn = PReluLayer(nn, name='act_prelu_last')

        with tf.variable_scope("part2"):
            for j in range(nb_block):
                with tf.variable_scope("block%d" %j):
                    if j==0:
                        nn = residual_block_dropout(nn, filters=128, is_train=is_train, reuse=reuse, tensor_shape='change')
                    else:
                        nn = residual_block_dropout(nn, filters=128, is_train=is_train, reuse=reuse)
            nn = Conv2d(nn, 128, (3,3), (2,2), act=None, padding='SAME',W_init=w_init, b_init=b_init, name='conv_last' )
            nn = BatchNormLayer(nn, act=None, is_train=is_train, gamma_init=g_init, name='bn_last' )
            nn = PReluLayer(nn, name='act_prelu_last')

        with tf.variable_scope("part3"):
            for j in range(nb_block):
                with tf.variable_scope("block%d" %j):
                    if j==0:
                        nn = residual_block_dropout(nn, filters=256, is_train=is_train, reuse=reuse, tensor_shape='change')
                    else:
                        nn = residual_block_dropout(nn, filters=256, is_train=is_train, reuse=reuse)
            nn = Conv2d(nn, 256, (3,3), (2,2), act=None, padding='SAME',W_init=w_init, b_init=b_init, name='conv_last' )
            nn = BatchNormLayer(nn, act=None, is_train=is_train, gamma_init=g_init, name='bn_last' )
            nn = PReluLayer(nn, name='act_prelu_last')
       
        # nn = ElementwiseLayer([nn, temp], tf.add, name='final_add')
        mean_pool = GlobalMeanPool2d(nn) #[None, 48, 48, 512] -> [None, 512]
        # mean_pool = DropoutLayer(mean_pool, keep=0.7, name='final_dropout')
        logit = DenseLayer(mean_pool, n_units=7, act=None, W_init=w_init, name='final_dense')

        return logit, logit.outputs