import numpy as np
import h5py
import scipy.io
import time
import sys
import argparse
import src.models.create_models as models

from keras import backend as K
from keras.preprocessing import sequence
from keras.optimizers import RMSprop

from src.models.keras_model_utils import save_model, get_model, train_model, test_model
from src.data.data_utils import ModelData
if (K.backend() == 'tensorflow'):
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf

TRUNCATE_DATASET_RATIO = 0.5
MAX_EPOCH = 70
BATCH_SIZE=100
# Temporary
TRAIN_SAMPLES=2200000
VALID_SAMPLES=8000
TEST_SAMPLES=455024

def print_method_header(method_name=None, *args):
    print ("%s(" % method_name, end='')
    if (len(args) > 0):
        print (args[0], end='')
    for arg in args[1:]:
        print (", %s" % arg, end='')
    print (")")

def replace_at_ind(tup=None, ind=None, val=None):
    return tup[:ind] + (val,) + tup[ind+1:]

def create_model(model_name=None, *args):
    """ Set the model that is being used by the name of the function """
    print ("\nCreating model from ", end='')
    print_method_header(model_name, *args)

    # If there are any 'None', turn them into None
    for i, arg in enumerate(args):
        if arg == 'None':
            args = replace_at_ind(args, ind=i, val=None)

    if (hasattr(models, model_name) == False):
        sys.exit('No such model in src/models/create_models')
    return getattr(models, model_name)(*args)

def print_date_time():
    print ('\nThe date is ' + time.strftime('%m/%d/%Y'))
    print ('The time is ' + time.strftime('%I:%M:%S %p') + '\n')

def main(argv):

    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Run a Keras model on genetic sequences '
        + 'to derive epigenetic mechanisms')

    parser.add_argument('model_name', metavar='MODEL', help="The name of the function in src/models/create_models to create a model with")
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
    model = create_model(args.model_name, *args.model_args)
    model.summary()

    print ('Saving models in json and yaml format to '
           + args.json_file + ' and  ' + args.yaml_file)
    print ('Saving weights to ' + args.weights_file
           + ' and epoch logs to ' + args.log_file)
    save_model(model, json_file=args.json_file, yaml_file=args.yaml_file)

    print ('Retrieving train, validation, and test data')
    # train, valid, test = get_data_tuples(shrink_size=(TRUNCATE_DATASET_RATIO, 1, 1))
    data = ModelData()
    train, valid, test = data.get_data_tuples_generator(shrink_size=(TRUNCATE_DATASET_RATIO, 1, 1), 
                                                        nb_samples=(TRAIN_SAMPLES, VALID_SAMPLES, TEST_SAMPLES),
                                                        batch_size=BATCH_SIZE)
    print_date_time()

    train_model(model,
                train=train, valid=valid,
                max_epoch=MAX_EPOCH,
                batch_size=BATCH_SIZE,
                nb_samples=(TRAIN_SAMPLES, VALID_SAMPLES),
                weights_file=args.weights_file,
                log_file=args.log_file,
                tensorboard_dir=args.tensorboard_dir)

    print_date_time()
    test_model(model, test=test, nb_samples=TEST_SAMPLES)

if __name__ == '__main__':
    main(sys.argv[1:])
