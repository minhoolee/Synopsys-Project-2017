#! /usr/bin/env python

"""Implement one dimeinsional wide residual network"""
# Copyright 2016 asmith26

from six.moves import range
import os

import sys
sys.stdout = sys.stderr
# Prevent reaching to maximum recursion depth in `theano.tensor.grad`
sys.setrecursionlimit(2 ** 20)

import numpy as np
np.random.seed(2 ** 10)

from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K

# nb_classes = 919

# Parameters from paper
# depth = 16              # table 5 on page 8 indicates best value (4.17) CIFAR-10
# k = 2                  # 'widen_factor'; table 5 on page 8 indicates best value (4.17) CIFAR-10
# # k = 4
# dropout_probability = 0.3 # table 6 on page 10 indicates best value (4.17) CIFAR-10
#
# weight_decay = 0.0005   # page 10: "Used in all experiments"
#
# batch_size = 128        # page 8: "Used in all experiments"
#
# use_bias = False
# weight_init = "he_normal"
# channel_axis = -1
# input_shape = (1000, 4)

k = 10  # 'widen_factor'; pg 8, table 5 indicates best value(4.00) CIFAR-10
depth = 16 # table 5 on page 8 indicates best value (4.00) CIFAR-10
dropout_probability = 0.3 # table 6 on page 10 indicates best value (3.89) CIFAR-10
weight_decay = 0.0005     # page 10: "Used in all experiments"
weight_init = 'he_normal'
channel_axis = -1
use_bias = False


# Wide residual network http://arxiv.org/abs/1605.07146
def _wide_basic(n_input_plane, n_output_plane, n_block, n_layer, stride):
    def f(net):
        # format of conv_params:
        #               [ [filter_length,
        #               subsample_length="stride",
        #               border_mode="same" or "valid"] ]
        # B(3,3): orignal <<basic>> block
        conv_params = [ [3, stride, "same"],
                        [3, 1, "same"] ] 

        n_bottleneck_plane = n_output_plane

        # Residual block
        for i, v in enumerate(conv_params):
            if i == 0:
                if n_input_plane != n_output_plane:
                    net = BatchNormalization(axis=channel_axis, name='wide_res_batchnorm1_block' + str(n_block) + '_' + str(i) + '_' + str(n_layer))(net)
                    net = Activation("relu", name='wide_res_relu1_block' + str(n_block) + '_' + str(i) + '_' + str(n_layer))(net)
                    convs = net
                else:
                    convs = BatchNormalization(axis=channel_axis, name='wide_res_batchnorm1_block' + str(n_block) + '_' + str(i) + '_' + str(n_layer))(net)
                    convs = Activation("relu", name='wide_res_relu1_block' + str(n_block) + '_' + str(i) + '_' + str(n_layer))(convs)
                convs = Convolution1D(n_bottleneck_plane, filter_length=v[0],
                                     subsample_length=v[1],
                                     border_mode=v[2],
                                     init=weight_init,
                                     W_regularizer=l2(weight_decay),
                                     bias=use_bias,
                                     name='wide_res_conv1_block' + str(n_block) + '_' + str(i) + '_' + str(n_layer))(convs)
            else:
                convs = BatchNormalization(axis=channel_axis, name='wide_res_batchnorm1_block' + str(n_block) + '_' + str(i) + '_' + str(n_layer))(convs)
                convs = Activation("relu", name='wide_res_relu1_block' + str(n_block) + '_' + str(i) + '_' + str(n_layer))(convs)
                if dropout_probability > 0:
                   convs = Dropout(dropout_probability, name='wide_res_dropout1_block' + str(n_block) + '_' + str(i) + '_' + str(n_layer))(convs)
                convs = Convolution1D(n_bottleneck_plane, filter_length=v[0],
                                     subsample_length=v[1],
                                     border_mode=v[2],
                                     init=weight_init,
                                     W_regularizer=l2(weight_decay),
                                     bias=use_bias,
                                     name='wide_res_conv1_block' + str(n_block) + '_' + str(i) + '_' + str(n_layer))(convs)

        # Shortcut Conntection: identity function or 1x1 convolutional
        #  (depends on difference between input & output shape - this
        #   corresponds to whether we are using the first block in each
        #   group; see _layer() ).
        if n_input_plane != n_output_plane:
            shortcut = Convolution1D(n_output_plane, filter_length=1,
                                     subsample_length=stride,
                                     border_mode="same",
                                     init=weight_init,
                                     W_regularizer=l2(weight_decay),
                                     bias=use_bias,
                                     name='wide_res_shortcut_block' + str(n_block) + '_' + str(i) + '_' + str(n_layer))(net)
        else:
            shortcut = net

        return merge([convs, shortcut], mode="sum") 

    return f


# "Stacking Residual Units on the same stage"
def _layer(block, n_input_plane, n_output_plane, n_block, count, stride):
    def f(net):
        net = block(n_input_plane, n_output_plane, n_block, 1, stride)(net)
        for i in range(2,int(count+1)):
            net = block(n_output_plane, n_output_plane, n_block, i, stride=1)(net)
        return net 

    return f


def create_model():
    assert((depth - 4) % 6 == 0)

    n = (depth - 4) / 6 
    inputs = Input(shape=input_shape)

    n_stages=[16, 16*k, 32*k, 64*k]

    conv1 = Convolution1D(nb_filter=n_stages[0], filter_length=3, 
                          subsample_length=1,
                          border_mode="same",
                          init=weight_init,
                          W_regularizer=l2(weight_decay),
                          bias=use_bias)(inputs) # "One conv at the beginning (spatial size: 32x32)"

    # Add wide residual blocks
    block_fn = _wide_basic
    conv2 = _layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], n_block=1, count=n, stride=1)(conv1)# "Stage 1 (spatial size: 32x32)"
    conv3 = _layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], n_block=2, count=n, stride=2)(conv2)# "Stage 2 (spatial size: 16x16)"
    conv4 = _layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], n_block=3, count=n, stride=2)(conv3)# "Stage 3 (spatial size: 8x8)"

    batch_norm = BatchNormalization(axis=channel_axis)(conv4)
    relu = Activation("relu")(batch_norm) 

    # Classifier block
    pool = AveragePooling1D(pool_length=8, stride=1, border_mode="same")(relu)
    flatten = Flatten()(pool)
    predictions = Dense(output_dim=nb_classes, init=weight_init, bias=use_bias,
                        W_regularizer=l2(weight_decay), activation="softmax")(flatten)

    model = Model(input=inputs, output=predictions)
    return model

