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

def VGG19(images, is_train=False, reuse=False):
    w_init, b_init, g_init = initialize_parameters()

    with tf.variable_scope("VGG19", reuse=reuse):
        nn = InputLayer(images, name='Input images')

        for i in range(2):
            nn = Conv2d(nn, 64, (3,3), (1,1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_%s' %i)
            nn = BatchNormLayer(nn, act=None, gamma_init=g_init, name='bn1_%s' %i)
            nn = PReluLayer(nn, name='prelu1_%s' %i)
        nn = MaxPool2d(nn, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='maxpool1')

        for i in range(2):
            nn = Conv2d(nn, 128, (3,3), (1,1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_%s' %i)
            nn = BatchNormLayer(nn, act=None, gamma_init=g_init, name='bn2_%s' %i)
            nn = PReluLayer(nn, name='prelu2_%s' %i)
        nn = MaxPool2d(nn, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='maxpool2')

        for i in range(4):
            nn = Conv2d(nn, 256, (3,3), (1,1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='conv3_%s' %i)
            nn = BatchNormLayer(nn, act=None, gamma_init=g_init, name='bn3_%s' %i)
            nn = PReluLayer(nn, name='prelu3_%s' %i)
        nn = MaxPool2d(nn, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='maxpool3')

        for i in range(4):
            nn = Conv2d(nn, 512, (3,3), (1,1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='conv4_%s' %i)
            nn = BatchNormLayer(nn, act=None, gamma_init=g_init, name='bn4_%s' %i)
            nn = PReluLayer(nn, name='prelu4_%s' %i)
        nn = MaxPool2d(nn, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='maxpool4')

        for i in range(4):
            nn = Conv2d(nn, 512, (3,3), (1,1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='conv5_%s' %i)
            nn = BatchNormLayer(nn, act=None, gamma_init=g_init, name='bn5_%s' %i)
            nn = PReluLayer(nn, name='prelu5_%s' %i)
        nn = MaxPool2d(nn, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='maxpool5')

        nn = GlobalMeanPool2d(nn, name='GlobalMeanPool')
        nn = DenseLayer(nn, n_units=1024, act=tf.nn.relu, name='fc6')
        nn = DenseLayer(nn, n_units=7, act=tf.identity, name='fc8')
        # nn = FlattenLayer(nn, name='flatten')
        # nn = DenseLayer(nn, n_units=1024, act=tf.nn.relu, name='fc6')
        # nn = DenseLayer(nn, n_units=1024, act=tf.nn.relu, name='fc7')
        # nn = DenseLayer(nn, n_units=7, act=tf.identity, name='fc8')

        return nn, nn.outputs
