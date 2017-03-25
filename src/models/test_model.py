import numpy as np
import h5py
import scipy.io
import time
import sys
import argparse
import src.models.create_models as models

from os.path import isfile
from keras import backend as K
from keras.preprocessing import sequence
from keras.optimizers import RMSprop

from src.models.keras_model_utils import Model
from src.data.data_utils import ModelData
from src.logging import log_utils

if (K.backend() == 'tensorflow'):
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf

TRUNCATE_DATASET_RATIO = 0.5

_log = log_utils.logger(__name__)

def main(argv):
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Test a Keras model on genetic sequences '
        + 'to derive epigenetic mechanisms')

    parser.add_argument('model_name', metavar='MODEL_NAME', help="The unique name of the model to create")
    # parser.add_argument('create_fn', metavar='MODEL_FUNC', help="The name of the function in src/models/create_models to create a model with")
    parser.add_argument('weights_file', metavar='WEIGHTS_FILE', help="The file (.hdf5) to store the model's weights")
    parser.add_argument('json_file', metavar='JSON_FILE', help="The file (.json) to store the model's architecture in JSON")
    parser.add_argument('yaml_file', metavar='YAML_FILE', help="The file (.yaml) to store the model's architecture in YAML")
    parser.add_argument('log_file', metavar='LOG_FILE', help="The file (.csv) to store the model's epoch logs")
    parser.add_argument('tensorboard_dir', metavar='TB_DIR', help="The directory to store the model's tensorboard data (if using Tensorflow backend)")
    args = parser.parse_args(argv)

    # Configure the tensorflow session to not run out of memory
    if (K.backend() == 'tensorflow'):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.85
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

    data = ModelData()

    model = Model(name=args.model_name)
    print(args.json_file)
    model.load_from(json_file=args.json_file, yaml_file=args.yaml_file) # Temporary solution to running a model under a new name
    model.load_weights(weights_file=args.weights_file)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    _log.info('Retrieving test data...')
    test = data.get_test_tuple()

    log_utils.print_date_time(_log)
    start = time.time()

    _log.info('Testing model...')
    model.test(test=test)
    log_utils.print_elapsed_time(_log, start=start, end=time.time())

    _log.info('Creating predictions...')
    y_predict = model.predict(test.X_test)
    log_utils.print_elapsed_time(_log, start=start, end=time.time())

    dict = {}
    dict['predictions'] = np.array(y_predict)
    scipy.io.savemat('models/predictions/y_predict_' + args.model_name + '.mat', dict)

    log_utils.print_date_time(_log)

if __name__ == '__main__':
    main(sys.argv[1:])
