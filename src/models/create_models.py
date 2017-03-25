"""
Different deep learning models whose architectures were tested

conv_net() is the final model that was used for the competition
"""

import keras.backend as K
from keras import layers, metrics, objectives
from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.pooling import GlobalAveragePooling1D, AveragePooling1D
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.optimizers import SGD, Adam
# from keras.applications.resnet50 import ResNet50
from keras.regularizers import l2
from src.models.resnet50 import ResNet50
from src.models.resnet50 import conv_block, identity_block
from src.models.wavenet_utils import CausalAtrousConvolution1D, categorical_mean_squared_error
from src.models import wide_res_net, wave_net
from src.logging import log_utils

_log = log_utils.logger(__name__)

def DanQ():
    """
    https://doi.org/10.1093/nar/gkw226
    """
    _log.info('Building the model')
    model = Sequential()
    model.add(Convolution1D(input_dim=4,
                            input_length=1000,
                            nb_filter=320,
                            filter_length=26,
                            border_mode="valid",
                            activation="relu",
                            subsample_length=1))

    model.add(MaxPooling1D(pool_length=13, stride=13))

    # model.add(Dropout(0.2))

    # model.add(Bidirectional(LSTM(input_dim=320, output_dim=320, 
    #                              dropout_W=0.2, dropout_U=0.5, 
    #                              return_sequences=True)))

    # model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(input_dim=320, output_dim=320,
                                dropout_W=0.2, dropout_U=0.5,
                                # activation='relu', 
                                return_sequences=True)))

    model.add(Flatten())

    model.add(Dense(input_dim=75*640, output_dim=925))
    model.add(Activation('relu'))

    model.add(Dense(input_dim=925, output_dim=919))
    model.add(Activation('sigmoid'))

    _log.info('Compiling model')
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model;

def conv_net():
    _log.info('Building the model')
    input = Input(shape=(1000, 4), name='input')

    # # ------------------------------------------------------------------------
    # # Wide Residual Layers http://arxiv.org/abs/1605.07146
    # # ------------------------------------------------------------------------
    # k = 10  # 'widen_factor'; pg 8, table 5 indicates best value(4.00) CIFAR-10
    # depth = 16 # table 5 on page 8 indicates best value (4.00) CIFAR-10
    # dropout_probability = 0.3 # table 6 on page 10 indicates best value (3.89) CIFAR-10
    # weight_decay = 0.0005     # page 10: "Used in all experiments"
    # weight_init = 'he_normal'
    # channel_axis = -1
    # use_bias = False
    # assert((depth - 4) % 6 == 0)
    # n = (depth - 4) / 6 
    # n = 2
    #
    # block_fn = wide_res_net._wide_basic
    # a = 16
    # n_stages = [a, a*k, 2*a*k, 4*a*k, 8*a*k]
    #
    # x = Convolution1D(nb_filter=n_stages[0], filter_length=3, 
    #                       subsample_length=1,
    #                       border_mode="same",
    #                       init=weight_init,
    #                       W_regularizer=l2(weight_decay),
    #                       bias=use_bias, name='wide_res_net_input_conv')(input) # "One conv at the beginning (spatial size: 32x32)"
    #
    # x = BatchNormalization(name='wide_res_net_input_batchnorm')(x)
    # x = Dropout(0.2)(x)
    # x = MaxPooling1D(pool_length=2, stride=2)(x)
    #
    # # "Stage 1 (spatial size: 32x32)"
    # x = wide_res_net._layer(block_fn, 
    #                       n_input_plane=n_stages[0], 
    #                       n_output_plane=n_stages[1], 
    #                       n_block=1,
    #                       count=n, stride=1)(x)
    #
    # x = Dropout(0.2)(x)
    # x = MaxPooling1D(pool_length=2, stride=2)(x)
    #
    # # "Stage 2 (spatial size: 16x16)"
    # x = wide_res_net._layer(block_fn, 
    #                       n_input_plane=n_stages[1], 
    #                       n_output_plane=n_stages[2], 
    #                       n_block=2,
    #                       count=n, stride=2)(x)
    #
    # # "Stage 3 (spatial size: 8x8)"
    # x = wide_res_net._layer(block_fn, 
    #                       n_input_plane=n_stages[2], 
    #                       n_output_plane=n_stages[3], 
    #                       n_block=3,
    #                       count=n, stride=2)(x)
    #
    # x = wide_res_net._layer(block_fn, 
    #                       n_input_plane=n_stages[3], 
    #                       n_output_plane=n_stages[4], 
    #                       n_block=4,
    #                       count=n, stride=2)(x)
    #
    # x = BatchNormalization(name='wide_res_net_batchnorm1')(x)
    # x = Activation('relu', name='wide_res_net_relu1')(x) 
    # # x = PReLU(name='wide_res_net_relu1')(x) 
    # x = AveragePooling1D(pool_length=5, stride=1, name='avg_pool')(x)
    # # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # Residual Layers
    # ------------------------------------------------------------------------
    # x = Convolution1D(nb_filter=64, filter_length=3, name='conv1')(input)
    # x = BatchNormalization(name='bn_conv1')(x)
    # x = Activation('relu')(x)
    # x = MaxPooling1D(pool_length=2, stride=2)(x)
    # x = Dropout(0.2)(x)
    #
    # x = conv_block(input, 3, [64, 64, 256], stage=2, block='a', stride=1)
    # x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    # x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    #
    # x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', stride=2)
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    #
    # x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    #
    # # x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    # # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    # # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    # #
    # x = AveragePooling1D(3, name='avg_pool')(x)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # Deep Convolutional Layers
    # ------------------------------------------------------------------------
    # Block 1
    x = Convolution1D(nb_filter=64, filter_length=3, init='he_normal', name='block1_conv1')(input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_length=2, stride=2, name='block1_pool1')(x)

    x = Convolution1D(nb_filter=64, filter_length=3, init='he_normal', name='block1_conv2')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_length=2, stride=2, name='block1_pool2')(x)

    # Block 2
    x = Convolution1D(nb_filter=128, filter_length=3, init='he_normal', name='block2_conv1')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Convolution1D(nb_filter=128, filter_length=3, init='he_normal', name='block2_conv2')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Dropout(0.2, name='block2_dropout1')(x)
    x = MaxPooling1D(pool_length=2, stride=2, name='block2_pool1')(x)

    # Block 3
    x = Convolution1D(nb_filter=256, filter_length=3, init='he_normal', name='block3_conv1')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Dropout(0.2, name='block3_dropout1')(x)
    x = MaxPooling1D(pool_length=2, stride=2, name='block3_pool1')(x)

    x = Convolution1D(nb_filter=256, filter_length=3, init='he_normal', name='block3_conv2')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Dropout(0.2, name='block3_dropout2')(x)
    x = MaxPooling1D(pool_length=2, stride=2, name='block3_pool2')(x)

    x = Convolution1D(nb_filter=512, filter_length=3, init='he_normal', name='block4_conv1')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Dropout(0.2, name='block4_dropout1')(x)
    x = MaxPooling1D(pool_length=2, stride=2, name='block4_pool1')(x)

    x = Convolution1D(nb_filter=512, filter_length=3, init='he_normal', name='block4_conv2')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # Maxpooling-Dropout https://arxiv.org/pdf/1512.01400
    # ------------------------------------------------------------------------
    x = Dropout(0.2, name='dropout1')(x)
    x = MaxPooling1D(pool_length=2, stride=2, name='dropout1_pool1')(x)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # Recurrent GRU Layers
    # ------------------------------------------------------------------------
    x = Bidirectional(GRU(input_dim=256, output_dim=256,
                          return_sequences=True, name='gru1'))(x)
    x = Dropout(0.5, name='gru_dropout1')(x)
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # Fully Connected Layers
    # ------------------------------------------------------------------------
    x = Flatten(name='flatten')(x)
    x = Dense(input_dim=2048, output_dim=1024, init='he_normal', name='fc1')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5, name='fc1_dropout')(x)
    x = Dense(input_dim=1024, output_dim=919, init='he_normal', name='fc2')(x)
    output = Activation('sigmoid', name='fc2_sigmoid')(x)
    # ------------------------------------------------------------------------

    model = Model(input, output)

    _log.info('Compiling model')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def _mult_conv_max(model=None, nb_conv=1, nb_filter=64):
    for i in range(nb_conv):
        model.add(Convolution1D(input_dim=4,
                                input_length=1000,
                                nb_filter=nb_filter,
                                filter_length=3,
                                border_mode='same',
                                # init='he_normal',
                                subsample_length=1))
        model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=2, stride=2))

def vgg_net():
    """
    https://arxiv.org/pdf/1409.1556.pdf
    """
    _log.info('Building the model')
    model = Sequential()
    _mult_conv_max(model, nb_conv=2, nb_filter=64)
    _mult_conv_max(model, nb_conv=2, nb_filter=128)
    _mult_conv_max(model, nb_conv=3, nb_filter=256)
    _mult_conv_max(model, nb_conv=3, nb_filter=512)
    _mult_conv_max(model, nb_conv=3, nb_filter=512)

    model.add(Flatten())
    model.add(Dense(output_dim=4000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=4000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=919))
    model.add(Activation('sigmoid'))

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    _log.info('Compiling model')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def vgg_net_16(weights='imagenet'):
    '''Instantiate the VGG16 architecture,
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
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
    # Returns
        A Keras model instance.
    '''

    TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5'
    TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either ' \
                         '`None` (random initialization) or `imagenet` ' \
                         '(pre-training on ImageNet).')

    _log.info('Building the model')
    # Block 1
    model = Sequential()
    model.add(Convolution1D(input_dim=4, input_length=1000, 
                            nb_filter=64, filter_length=3, activation='relu', border_mode='same', name='block1_conv1'))
    model.add(Convolution1D(nb_filter=64, filter_length=3, activation='relu', border_mode='same', name='block1_conv2'))
    model.add(MaxPooling1D(pool_length=2, stride=2, name='block1_pool'))

    # Block 2
    model.add(Convolution1D(nb_filter=128, filter_length=3,activation='relu', border_mode='same', name='block2_conv1'))
    model.add(Convolution1D(nb_filter=128, filter_length=3, activation='relu', border_mode='same', name='block2_conv2'))
    model.add(MaxPooling1D(pool_length=2, stride=2, name='block2_pool'))

    # Block 3
    model.add(Convolution1D(nb_filter=256, filter_length=3, activation='relu', border_mode='same', name='block3_conv1'))
    model.add(Convolution1D(nb_filter=256, filter_length=3, activation='relu', border_mode='same', name='block3_conv2'))
    model.add(Convolution1D(nb_filter=256, filter_length=3, activation='relu', border_mode='same', name='block3_conv3'))
    model.add(MaxPooling1D(pool_length=2, stride=2, name='block3_pool'))

    # Block 4
    model.add(Convolution1D(nb_filter=512, filter_length=3, activation='relu', border_mode='same', name='block4_conv1'))
    model.add(Convolution1D(nb_filter=512, filter_length=3, activation='relu', border_mode='same', name='block4_conv2'))
    model.add(Convolution1D(nb_filter=512, filter_length=3, activation='relu', border_mode='same', name='block4_conv3'))
    model.add(MaxPooling1D(pool_length=2, stride=2, name='block4_pool'))

    # Block 5
    model.add(Convolution1D(nb_filter=512, filter_length=3, activation='relu', border_mode='same', name='block5_conv1'))
    model.add(Convolution1D(nb_filter=512, filter_length=3, activation='relu', border_mode='same', name='block5_conv2'))
    model.add(Convolution1D(nb_filter=512, filter_length=3, activation='relu', border_mode='same', name='block5_conv3'))
    model.add(MaxPooling1D(pool_length=2, stride=2, name='block5_pool'))

    # Classification block
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dense(4096, activation='relu', name='fc2'))
    # model.add(Dense(1000, activation='softmax', name='predictions'))

    # load weights
    if weights == 'imagenet':
        _log.info('K.image_dim_ordering:', K.image_dim_ordering())
        if K.image_dim_ordering() == 'th':
            weights_path = get_file('vgg_net_16.h5',
                                    TH_WEIGHTS_PATH,
                                    cache_subdir='models')
            model.load_weights(weights_path)
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            weights_path = get_file('vgg_net_16.h5',
                                    TF_WEIGHTS_PATH,
                                    cache_subdir='models')
            model.load_weights(weights_path)
            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)

    _log.info('Compiling model')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def ResNet():
    """
    [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
    """
    _log.info('Building the model')
    # if K.image_dim_ordering() == 'th':
    input = Input(shape=(1000, 4)) # (channels, width, height)
    # elif K.image_dim_ordering() == 'tf':
    #     input = Input(shape=(1000, 4, 1)) # (width, height, channels)
    pretrained_model = ResNet50(include_top=False, input_tensor=input)

    x = pretrained_model.output
    # x = GlobalAveragePooling1D()(x)
    x = Flatten()(x)
    predictions = Dense(919, activation='softmax', name='fc1')(x)
    model = Model(input=pretrained_model.input, output=predictions) 

    # Freeze pretrained ResNet50
    for layer in pretrained_model.layers:
        layer.trainable = False

    _log.info('Compiling model')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def WideResNet():
    """
    http://arxiv.org/abs/1605.07146
    """
    _log.info('Building the model')
    model = wide_res_net.create_model()

    _log.info('Compiling model')
    sgd = SGD(lr=0.1, momentum=0.8, nesterov=True)

    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

    return model

def WaveNet():
    """
    https://arxiv.org/abs/1609.03499
    """
    desired_sample_rate = 4410
    dilation_depth = 8  #
    nb_stacks = 1

    # The temporal-first outputs are computed from zero-padding. Setting below to True ignores these inputs:
    train_only_in_receptive_field = True

    _log.info('Building model...')
    model = wave_net.create_model(desired_sample_rate, dilation_depth, nb_stacks)

    _log.info('Compiling model...')
    # loss = objectives.categorical_crossentropy
    loss = objectives.binary_crossentropy
    all_metrics = [
        # metrics.categorical_accuracy,
        # wave_net.categorical_mean_squared_error,
        'accuracy'
    ]
    # if train_only_in_receptive_field:
    #     loss = wave_net.skip_out_of_receptive_field(loss, desired_sample_rate, dilation_depth, nb_stacks)
    #     all_metrics = [wave_net.skip_out_of_receptive_field(m, desired_sample_rate, dilation_depth, nb_stacks) for m in all_metrics]
    adam = Adam(lr=3e-4)
    # model.compile(optimizer='adam', loss=loss, metrics=all_metrics)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def WaveNetMinimal():
    """
    Repurposed for my use from https://arxiv.org/abs/1609.03499
    """
    dilation_depth = 8  #
    nb_stacks = 1
    nb_output_bins = 4
    nb_filters = 64
    use_bias = False

    def residual_block(x):
        original_x = x
        # TODO: initalization, regularization?
        # Note: The AtrousConvolution1D with the 'causal' flag is implemented in github.com/basveeling/keras#@wavenet.
        tanh_out = CausalAtrousConvolution1D(nb_filters, 2, atrous_rate=2 ** i, border_mode='valid', causal=True,
                                             bias=use_bias,
                                             name='dilated_conv_%d_tanh_s%d' % (2 ** i, s), activation='tanh')(x)
        x = layers.Dropout(0.2)(x)
        sigm_out = CausalAtrousConvolution1D(nb_filters, 2, atrous_rate=2 ** i, border_mode='valid', causal=True,
                                             bias=use_bias,
                                             name='dilated_conv_%d_sigm_s%d' % (2 ** i, s), activation='sigmoid')(x)
        x = layers.Merge(mode='mul', name='gated_activation_%d_s%d' % (i, s))([tanh_out, sigm_out])

        res_x = layers.Convolution1D(nb_filters, 1, border_mode='same', bias=use_bias)(x)
        res_x = layers.Merge(mode='sum')([original_x, res_x])
        return res_x

    _log.info('Building model...')
    input = Input(shape=(1000, nb_output_bins), name='input_part')
    out = input
    out = CausalAtrousConvolution1D(nb_filters, 2, atrous_rate=1, border_mode='valid', causal=True,
                                    name='initial_causal_conv')(out)

    for s in range(nb_stacks):
        for i in range(0, dilation_depth + 1):
            out = residual_block(out)

    out = layers.PReLU()(out)
    out = layers.Convolution1D(nb_filter=64, filter_length=3, border_mode='same', init='he_normal')(out)
    out = layers.Dropout(0.2)(out)
    out = layers.Activation('relu')(out)
    out = layers.Convolution1D(nb_filter=64, filter_length=3, border_mode='same', init='he_normal')(out)
    out = layers.Dropout(0.2)(out)
    out = layers.Activation('relu')(out)

    out = layers.Flatten()(out)
    predictions = layers.Dense(919, name='fc1')(out)
    predictions = layers.Activation('sigmoid', name="output_sigmoid")(predictions)
    model = Model(input, predictions)

    _log.info('Compiling model...')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
