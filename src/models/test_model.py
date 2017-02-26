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

from src.models.keras_model_utils import save_model, get_model, train_model, test_model
from src.data.data_utils import ModelData

if (K.backend() == 'tensorflow'):
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf

TRUNCATE_DATASET_RATIO = 0.5

def print_date_time():
    print ('\nThe date is ' + time.strftime('%m/%d/%Y'))
    print ('The time is ' + time.strftime('%I:%M:%S %p') + '\n')

def main(argv):

    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Test a Keras model on genetic sequences '
        + 'to derive epigenetic mechanisms')

    parser.add_argument('model_name', metavar='MODEL', help="The name of the function in src/models/create_models to create a model with")
    parser.add_argument('weights_file', metavar='WEIGHTS_FILE', help="The file (.hdf5) to store the model's weights")
    parser.add_argument('json_file', metavar='JSON_FILE', help="The file (.json) to store the model's architecture in JSON")
    parser.add_argument('yaml_file', metavar='YAML_FILE', help="The file (.yaml) to store the model's architecture in YAML")
    parser.add_argument('log_file', metavar='LOG_FILE', help="The file (.csv) to store the model's epoch logs")
    parser.add_argument('tensorboard_dir', metavar='TB_DIR', help="The directory to store the model's tensorboard data (if using Tensorflow backend)")
    args = parser.parse_args()

    # Configure the tensorflow session to not run out of memory
    if (K.backend() == 'tensorflow'):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.85
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

    print ('Retrieving train, validation, and test data')
    data = ModelData()
    train, valid, test = data.get_data_tuples(shrink_size=(TRUNCATE_DATASET_RATIO, 1, 1))

    model = get_model(json_file=args.json_file, yaml_file=args.yaml_file)
    if (not isfile(args.weights_file)):
        sys.exit('No file that contains the weights of the model to test')
    print ('Loading weights from ' + args.weights_file)
    model.load_weights(args.weights_file)

    # Temporary solution to Keras not saving compilation,
    # only works if model was compiled with default 'Adam'
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    print_date_time()

    test_model(model, test=test)
    print_date_time()

if __name__ == '__main__':
    main(sys.argv[1:])