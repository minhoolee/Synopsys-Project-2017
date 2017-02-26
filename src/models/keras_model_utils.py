import json
import yaml
import sys
import os
from os.path import isfile
from shutil import copyfile
from types import GeneratorType

from keras import backend as K
from keras.models import load_model, model_from_json, model_from_yaml
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, ReduceLROnPlateau

from heraspy.callback import HeraCallback
from src.data.data_utils import Dataset, DataFile

class BatchCSVLogger(CSVLogger):
    """ Callback that streams batch results to a csv file.

    Stream losses and accuracies of the end of each batch

    # Example
        ```python
            batch_logger = BatchCSVLogger('training.log')
            model.fit(X_train, Y_train, callbacks=[batch_logger])
        ```
    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def on_train_begin(self, logs={}):
        super().on_train_begin(logs)
        self.batch_history = {}

    def on_batch_end(self, batch, logs={}):
        self.batch_history['loss'] = logs.get('loss')
        self.batch_history['acc'] = logs.get('acc')
        super().on_epoch_end(batch, logs=self.batch_history)

    def on_epoch_end(self, epoch, logs=None):
        pass

def save_model(model, json_file=None, yaml_file=None):
    """ Save Keras model in JSON and YAML format """
    if (json_file is None and yaml_file is None):
        sys.exit('No JSON or YAML files provided to save the model')

    if (json_file is not None):
        if (isfile(json_file)):
            print('Saving ' + json_file + ' to ' + json_file + '.old')
            copyfile(json_file, json_file + '.old')
        # Write model to JSON file
        with open(json_file, 'w') as outfile:
            outfile.write(json.dumps(json.loads(model.to_json()), indent=2))

    if (yaml_file is not None):
        if (isfile(yaml_file)):
            print('Saving ' + yaml_file + ' to ' + yaml_file + '.old')
            copyfile(yaml_file, yaml_file + '.old')
        # Write model to YAML file
        with open(yaml_file, 'w') as outfile:
            outfile.write(model.to_yaml())

def get_model(json_file=None, yaml_file=None):
    """ Return a Keras model from a JSON or YAML file """
    if (json_file is None and yaml_file is None):
        sys.exit('No JSON or YAML files provided to save the model')

    if (json_file is not None and isfile(json_file)):
        print('Opening model from ' + json_file)
        with open(json_file, 'r') as infile:
            json_string = json.load(infile)

            # Process the JSON file whether the model 
            # was written as a string or dict
            if (isinstance(json_string, dict)):
                model = model_from_json(json.dumps(json_string))
            elif (isinstance(json_string, str)):
                model = model_from_json(json_string)
            else:
                sys.exit(yaml_file + ' is not formatted in the format Keras uses')
            return model

    if (yaml_file is not None and isfile(yaml_file)):
        print('Opening model from ' + yaml_file)
        with open(yaml_file, 'r') as infile:
            yaml_string = yaml.load(infile)

            # Process the YAML file whether the model 
            # was written as a string or dict
            if (isinstance(yaml_string, dict)):
                model = model_from_yaml(yaml.dump(yaml_string))
            elif (isinstance(yaml_string, str)):
                model = model_from_yaml(yaml_string)
            else:
                sys.exit(yaml_file + ' is not formatted in the format Keras uses')
            return model

def train_model(model,
                train, valid,
                max_epoch, batch_size,
                nb_samples=None,
                weights_file=None,
                log_file=None,
                tensorboard_dir=None):

    if (isfile(weights_file)):
        print ('Loading weights from ' + weights_file)
        model.load_weights(weights_file)
        print ('Saving ' + weights_file + ' to ' + weights_file + '.old')
        copyfile(weights_file, weights_file + '.old')

    if (isfile(log_file)):
        print ('Saving ' + log_file + ' to ' + log_file + '.old')
        copyfile(log_file, log_file + '.old')

    # Build assisting callbacks for printing progress and stopping if no performance improvements
    checkpointer = ModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    csvlogger = CSVLogger(log_file)
    batchlogger = BatchCSVLogger(log_file.rsplit('.',1)[0] +"_batch." + log_file.rsplit('.',1)[1])
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-5)
    callbacks = [checkpointer, earlystopper, csvlogger, batchlogger, reduceLR]

    # heraslogger = HeraCallback(
    #     'model-key',
    #     'localhost',
    #     6000
    # )
    # callbacks.append(heraslogger)

    if (K.backend() == 'tensorflow'):
        print('Run `tensorboard --logdir=' + tensorboard_dir 
              + '` to open tensorboard at (default) 127.0.0.1:6006')
        tensorboard = TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)
        callbacks.append(tensorboard)

    print('Running at most ' + str(max_epoch) + ' epochs')

    if isinstance(valid, GeneratorType):
        # Validation data is to be passed from generator
        validation_data=valid
    else:
        # Validation data is in numpy arrays
        validation_data=(valid.X_valid, valid.y_valid)

    # try:
    if isinstance(train, GeneratorType):
        # Training data is to be passed from generator
        model.fit_generator(generator=train,
                            samples_per_epoch=nb_samples[Dataset.TRAIN.value],
                            nb_epoch=max_epoch, 
                            validation_data=validation_data,
                            nb_val_samples=nb_samples[Dataset.VALID.value],
                            callbacks=callbacks)
    else:
        # Training data is in numpy arrays
        model.fit(x=train.X_train, y=train.y_train, 
                  batch_size=batch_size, nb_epoch=max_epoch, shuffle=True, 
                  validation_data=validation_data,
                  callbacks=callbacks)
    # except KeyboardInterrupt:
        # TODO: improve this structure so python is the writing to all files
        # os.system("cat .tmp | col -b  >> ./models/run_logs/SAVED.txt")


def test_model(model, test, nb_samples=None):
    """
    Evaluates model's performance on test set
    """

    if isinstance(test, GeneratorType):
        tresults = model.evaluate_generator(generator=test,
                                            val_samples=nb_samples)
    else:
        tresults = model.evaluate(test.X_test, test.y_test)

    print(tresults)
