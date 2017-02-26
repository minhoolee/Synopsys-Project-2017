# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function

import numpy as np
import warnings

from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D, ZeroPadding1D, AveragePooling1D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.layers.core import Dropout
# from imagenet_utils import decode_predictions, preprocess_input


TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = -1
    # if K.image_dim_ordering() == 'tf':
    #     bn_axis = 3
    # else:
    #     bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution1D(nb_filter1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution1D(nb_filter2, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution1D(nb_filter3, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, stride=2):
    '''conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample_length=2
    And the shortcut should have subsample_length=2 as well
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = -1
    # if K.image_dim_ordering() == 'tf':
    #     bn_axis = 3
    # else:
    #     bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution1D(nb_filter1, 1, subsample_length=stride,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution1D(nb_filter2, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Convolution1D(nb_filter3, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution1D(nb_filter3, 1, subsample_length=stride,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None):
    '''Instantiate the ResNet50 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    # if K.image_dim_ordering() == 'th':
    #     if include_top:
    #         input_shape = (3, 224)
    #     else:
    #         input_shape = (3, None)
    # else:
    #     if include_top:
    #         input_shape = (224, 3)
    #     else:
    #         input_shape = (None, 3)
    #
    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor)
    #     else:
    #         img_input = input_tensor
    img_input = input_tensor
    bn_axis = -1
    # if K.image_dim_ordering() == 'tf':
    #     bn_axis = 3
    # else:
    #     bn_axis = 1

    x = ZeroPadding1D(3)(img_input)
    x = Convolution1D(64, 7, subsample_length=2, name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3, stride=2)(x)
    x = Dropout(0.2)(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', stride=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling1D(7, name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(1000, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x)

    # load weights
    # if weights == 'imagenet':
    #     print('K.image_dim_ordering:', K.image_dim_ordering())
    #     if K.image_dim_ordering() == 'th':
    #         if include_top:
    #             weights_path = get_file('resnet50_weights_th_dim_ordering_th_kernels.h5',
    #                                     TH_WEIGHTS_PATH,
    #                                     cache_subdir='models',
    #                                     md5_hash='1c1f8f5b0c8ee28fe9d950625a230e1c')
    #         else:
    #             weights_path = get_file('resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
    #                                     TH_WEIGHTS_PATH_NO_TOP,
    #                                     cache_subdir='models',
    #                                     md5_hash='f64f049c92468c9affcd44b0976cdafe')
    #         model.load_weights(weights_path)
    #         if K.backend() == 'tensorflow':
    #             warnings.warn('You are using the TensorFlow backend, yet you '
    #                           'are using the Theano '
    #                           'image dimension ordering convention '
    #                           '(`image_dim_ordering="th"`). '
    #                           'For best performance, set '
    #                           '`image_dim_ordering="tf"` in '
    #                           'your Keras config '
    #                           'at ~/.keras/keras.json.')
    #             convert_all_kernels_in_model(model)
    #     else:
    #         if include_top:
    #             weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
    #                                     TF_WEIGHTS_PATH,
    #                                     cache_subdir='models',
    #                                     md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
    #         else:
    #             weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                                     TF_WEIGHTS_PATH_NO_TOP,
    #                                     cache_subdir='models',
    #                                     md5_hash='a268eb855778b3df3c7506639542a6af')
    #         model.load_weights(weights_path)
    #         if K.backend() == 'theano':
    #             convert_all_kernels_in_model(model)
    return model
