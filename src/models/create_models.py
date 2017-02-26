import keras.backend as K
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.pooling import GlobalAveragePooling1D
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.optimizers import SGD, Adam
# from keras.applications.resnet50 import ResNet50
from src.models.resnet50 import ResNet50
from src.models import wide_res_net
from keras.layers import Input

def DanQ():
    print ('Building the model')
    model = Sequential()
    model.add(Convolution1D(input_dim=4,
                            input_length=1000,
                            nb_filter=320,
                            filter_length=26,
                            border_mode="valid",
                            activation="relu",
                            subsample_length=1))

    model.add(MaxPooling1D(pool_length=13, stride=13))

    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(input_dim=320, output_dim=320, return_sequences=True)))

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(input_dim=75*640, output_dim=925))
    model.add(Activation('relu'))

    model.add(Dense(input_dim=925, output_dim=919))
    model.add(Activation('sigmoid'))

    print ('Compiling model')
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model;

def conv_net():
    print ('Building the model')
    model = Sequential()
    model.add(Convolution1D(input_dim=4, input_length=1000, 
                            nb_filter=64, filter_length=3, init='he_normal'))
    # model.add(Activation('relu'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Convolution1D(nb_filter=64, filter_length=3, init='he_normal'))
    # model.add(Activation('relu'))
    model.add(PReLU())
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # model.add(MaxPooling1D(pool_length=2, stride=2))

    model.add(Convolution1D(nb_filter=128, filter_length=3, init='he_normal', subsample_length=2))
    # model.add(Activation('relu'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Convolution1D(nb_filter=128, filter_length=3, init='he_normal', subsample_length=2))
    # model.add(Activation('relu'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # model.add(MaxPooling1D(pool_length=2, stride=2))

    # model.add(Convolution1D(nb_filter=256, filter_length=3, init='he_normal'))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    model.add(Convolution1D(nb_filter=256, filter_length=5, init='he_normal', subsample_length=2))
    # model.add(Activation('relu'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Convolution1D(nb_filter=256, filter_length=5, init='he_normal', subsample_length=2))
    # model.add(Activation('relu'))
    model.add(PReLU())
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # model.add(MaxPooling1D(pool_length=2, stride=2))
    #
    # model.add(Convolution1D(nb_filter=512, filter_length=3, init='he_normal'))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # model.add(Convolution1D(nb_filter=512, filter_length=3, init='he_normal'))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # model.add(Convolution1D(nb_filter=512, filter_length=3, init='he_normal'))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # model.add(Convolution1D(nb_filter=512, filter_length=3, init='he_normal'))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))

    # model.add(MaxPooling1D(pool_length=2, stride=2))
    model.add(Flatten())
    model.add(Dense(input_dim=4096, output_dim=1024, init='he_normal'))
    # model.add(Activation('relu'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=1024, output_dim=919, init='he_normal'))
    # model.add(Activation('softmax'))
    # model.add(Convolution1D(nb_filter=919, filter_length=119, init='he_normal'))
    # model.add(Flatten())
    model.add(Activation('sigmoid'))

    print ('Compiling model')
    adam = Adam(lr=3e-4)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

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
    print ('Building the model')
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
    print ('Compiling model')
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

    print ('Building the model')
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
    model.add(Dense(1000, activation='softmax', name='predictions'))

    # load weights
    if weights == 'imagenet':
        print('K.image_dim_ordering:', K.image_dim_ordering())
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

    print ('Compiling model')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def ResNet():
    print ('Building the model')
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

    print ('Compiling model')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def WideResNet():
    print('Building the model')
    model = wide_res_net.create_model()

    print('Compiling model')
    sgd = SGD(lr=0.1, momentum=0.8, nesterov=True)

    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

    return model

