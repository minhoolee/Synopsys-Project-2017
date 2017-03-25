import numpy as np
import h5py
import scipy.io
import time
import sys
import argparse

# from datetime import timedelta

from keras import backend as K
from keras.preprocessing import sequence
from keras.optimizers import RMSprop

if (K.backend() == 'tensorflow'):
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf

from src.models.keras_model_utils import Model
from src.data.data_utils import ModelData
from src.logging import log_utils

_log = log_utils.logger(__name__)

#TRUNCATE_TRAIN_RATIO = 0.01
TRUNCATE_TRAIN_RATIO = 1
MAX_EPOCH = 70
BATCH_SIZE = 400
# BATCH_SIZE = 100
# Temporary
TRAIN_SAMPLES = 4400000 * TRUNCATE_TRAIN_RATIO 
VALID_SAMPLES = 8000
TEST_SAMPLES = 455024

def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

def main(argv):

    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Run a Keras model on genetic sequences '
        + 'to derive epigenetic mechanisms')

    parser.add_argument('model_name', metavar='MODEL_NAME', help="The unique name of the model to create")
    parser.add_argument('create_fn', metavar='MODEL_FUNC', help="The name of the function in src/models/create_models to create a model with")
    parser.add_argument('weights_file', metavar='WEIGHTS_FILE', help="The file (.hdf5) to store the model's weights")
    parser.add_argument('json_file', metavar='JSON_FILE', help="The file (.json) to store the model's architecture in JSON")
    parser.add_argument('yaml_file', metavar='YAML_FILE', help="The file (.yaml) to store the model's architecture in YAML")
    parser.add_argument('log_file', metavar='LOG_FILE', help="The file (.csv) to store the model's epoch logs")
    parser.add_argument('tensorboard_dir', metavar='TB_DIR', help="The directory to store the model's tensorboard data (if using Tensorflow backend)")
    parser.add_argument('--arg', dest='model_args', action='append', help="Optional arguments to be passed to create the model")
    args = parser.parse_args()

    # Configure the tensorflow session to not run out of memory
    if (K.backend() == 'tensorflow'):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

    # Create the model using the optional parameters passed in
    if (not args.model_args):
        args.model_args = []

    model = Model(name=args.model_name)
    model.create_from(args.create_fn, *args.model_args)

    # model.load_from('models/json/conv_net_large_res_5.json') # Temporary solution to running a model under a new name
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    _log.info('\n')
    # _log.info('Saving model to file system...')
    # model.save_to(json_file=args.json_file, yaml_file=args.yaml_file)

    _log.info('Loading model weights...')
    model.load_weights(weights_file=args.weights_file, by_name=True)
    # model.load_weights(weights_file='models/weights/danq_17.hdf5', by_name=True)
    # pop_layer(model)
    # model.layers.pop(); # Get rid of fc2 layer
    # model.outputs = [model.layers[-1].output]
    # model.output_layers = [model.layers[-1]]
    # model.layers[-1].outbound_notes = []


    data = ModelData(batch_size=BATCH_SIZE)
    # Shrink the training dataset to half of its original size
    # train, valid, test = data.get_data_tuples_generator(shrink_size=(TRUNCATE_TRAIN_RATIO, 1, 1), 
    #                                                     nb_samples=(TRAIN_SAMPLES, VALID_SAMPLES, TEST_SAMPLES))
    # train, valid, test = data.get_data_tuples(shrink_size=(TRUNCATE_TRAIN_RATIO, 1, 1))
    _log.info('Retrieving training data...')
    train = data.get_train_tuple(shrink_size=TRUNCATE_TRAIN_RATIO)

    _log.info('Retrieving validation data...')
    valid = data.get_valid_tuple()

    log_utils.print_date_time(_log)
    _log.info('\n')
    start = time.time()

    _log.info('Training model...')
    model.train(train=train, valid=valid,
                weights_file=args.weights_file,
                max_epoch=MAX_EPOCH,
                batch_size=BATCH_SIZE,
                nb_samples=(TRAIN_SAMPLES, VALID_SAMPLES),
                log_file=args.log_file,
                tensorboard_dir=args.tensorboard_dir)

    _log.info('\n')
    log_utils.print_date_time(_log)
    log_utils.print_elapsed_time(_log, start=start, end=time.time())
    _log.info('\n')

    _log.info('Retrieving testing data...')
    test = data.get_test_tuple()

    _log.info('\n')
    _log.info('Testing model...')
    model.test(test=test, nb_samples=TEST_SAMPLES)

    _log.info('\n')
    _log.info('Creating predictions...')
    model.predict(test.X_test)

if __name__ == '__main__':
    main(sys.argv[1:])
