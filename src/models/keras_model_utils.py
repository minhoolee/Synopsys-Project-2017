import json
import yaml
import sys
import os
import logging
from os.path import isfile
from shutil import copyfile
from types import GeneratorType

from keras import backend as K
from keras.models import load_model, model_from_json, model_from_yaml
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, ReduceLROnPlateau

from src.data.data_utils import Dataset, DataFile
from src.logging import log_utils
import src.models.create_models as models

_log = log_utils.logger(__name__)

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

class Model():
    """
    Class that encapsulates a Keras model and can train, test, and analyze it
    """
    def __init__(self, 
                 name="model_default",
                 model=None):
        """
        # Arguments
            name: name of model
            model: optional instance of pre-created model
        """

        self.name = name
        self.model = model

    def _clean(self):
        _log.critical('\nSaving logs and shutting down...')
        os.system("cat .tmp | col -b  >> ./models/run_logs/" + self.name + ".txt")
        logging.shutdown()
        sys.exit()

    def summary(self):
        if self.model is None:
            _log.exception('Model has not been created yet')
            logging.shutdown()
        self.model.summary()

    def set_to(self, model):
        if model is None:
            _log.exception('Supplied model is None')
            logging.shutdown()
        else:
            _log.info('Setting model to the supplied one...')
            self.model = model

    def compile(self, 
                optimizer, 
                loss, 
                metrics=None, 
                loss_weights=None, 
                sample_weight_mode=None):

        if self.model is None:
            _log.exception('Model has not been created yet')
            logging.shutdown()
        _log.info('Compiling model')
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, 
                           loss_weights=loss_weights, sample_weight_mode=sample_weight_mode)

    def save_to(self, json_file=None, yaml_file=None):
        """ Save Keras model in JSON and YAML format """
        if (json_file is None and yaml_file is None):
            _log.exception('No JSON or YAML files provided to save the model')
            logging.shutdown()

        if (json_file is not None):
            if (isfile(json_file)):
                _log.info('Saving ' + json_file + ' to ' + json_file + '.old...')
                copyfile(json_file, json_file + '.old')

            _log.info('Saving model to ' + json_file)
            # Write model to JSON file
            with open(json_file, 'w') as outfile:
                outfile.write(json.dumps(json.loads(self.model.to_json()), indent=2))

        if (yaml_file is not None):
            if (isfile(yaml_file)):
                _log.info('Saving ' + yaml_file + ' to ' + yaml_file + '.old...')
                copyfile(yaml_file, yaml_file + '.old')

            _log.info('Saving model to ' + yaml_file)
            # Write model to YAML file
            with open(yaml_file, 'w') as outfile:
                outfile.write(self.model.to_yaml())


    def create_from(self, create_fn=None, *args):
        """ 
        Set the model that is being used by the name of the function 

        TODO: improve these parameters
        """
        hdr = log_utils.method_header(create_fn, *args)
        _log.info('Creating model from {:s}'.format(hdr) + '...')

        # If there are any 'None', turn them into None
        for i, arg in enumerate(args):
            if arg == 'None':
                args = log_utils.replace_at_ind(args, ind=i, val=None)

        if (hasattr(models, create_fn) == False):
            _log.exception('No such model in src/models/create_models')
            logging.shutdown()

        # Call the function with the supplied arguments to create the model
        self.model = getattr(models, create_fn)(*args)

    def load_from(self, json_file=None, yaml_file=None):
        """ Return a Keras model from a JSON or YAML file """
        if (json_file is None and yaml_file is None):
            _log.exception('No JSON or YAML files provided to save the model')
            logging.shutdown()

        if (json_file is not None and isfile(json_file)):
            _log.info('Loading model from ' + json_file + '...')
            with open(json_file, 'r') as infile:
                json_string = json.load(infile)

                # Process the JSON file whether the model 
                # was written as a string or dict
                if (isinstance(json_string, dict)):
                    self.model = model_from_json(json.dumps(json_string))
                elif (isinstance(json_string, str)):
                    self.model = model_from_json(json_string)
                else:
                    _log.exception(yaml_file + ' is not formatted in the format Keras uses')
                    logging.shutdown()

        if (yaml_file is not None and isfile(yaml_file)):
            _log.info('Loading model from ' + yaml_file + '...')
            with open(yaml_file, 'r') as infile:
                yaml_string = yaml.load(infile)

                # Process the YAML file whether the model 
                # was written as a string or dict
                if (isinstance(yaml_string, dict)):
                    self.model = model_from_yaml(yaml.dump(yaml_string))
                elif (isinstance(yaml_string, str)):
                    self.model = model_from_yaml(yaml_string)
                else:
                    _log.exception(yaml_file + ' is not formatted in the format Keras uses')
                    logging.shutdown()

        return self.model

    def load_weights(self, weights_file=None, by_name=False, exclude=[]):
        # TODO: allow loading latest weights and best weights
        if isfile(weights_file):
            _log.info('Loading weights from ' + weights_file + '...')
            if not exclude:
                self.model.load_weights(weights_file, by_name)
            else:
                # TODO: Figure this out
                # layer_dict = dict[(l.name, l) for l in self.model.layers]
                # for l in layer_dict:
                #     l.set_weights()
                print('Not supported yet, please change name of layer')

    def train(self,
              train, valid,
              weights_file,
              max_epoch, batch_size,
              nb_samples=None,
              log_file=None,
              tensorboard_dir=None):

        if isfile(log_file):
            _log.info('Saving ' + log_file + ' to ' + log_file + '.old...')
            copyfile(log_file, log_file + '.old')

        # Build assisting callbacks for printing progress and stopping if no performance improvements
        bestcheckpointer = ModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=True)
        checkpointer = ModelCheckpoint(filepath=os.path.splitext(weights_file)[0] 
                                       + '.{epoch:02d}-{val_loss:.3f}.hdf5', 
                                       verbose=1, save_best_only=False)
        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        csvlogger = CSVLogger(log_file)
        batchlogger = BatchCSVLogger(log_file.rsplit('.',1)[0] +"_batch." + log_file.rsplit('.',1)[1])
        # reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-5)
        callbacks = [bestcheckpointer, checkpointer, earlystopper, csvlogger, batchlogger]

        # Keras currently will fail with OOM for tensorboard if validation data is not None
        if (K.backend() == 'tensorflow' and valid is None):
            _log.info('Run `tensorboard --logdir=' + tensorboard_dir 
                  + '` to open tensorboard at (default) 127.0.0.1:6006')
            tensorboard = TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)
            callbacks.append(tensorboard)

        if isinstance(valid, GeneratorType):
            # Validation data is to be passed from generator
            validation_data=valid
        else:
            # Validation data is in numpy arrays
            validation_data=(valid.X_valid, valid.y_valid)
        try:
            _log.info('Running at most ' + str(max_epoch) + ' epochs')
            _log.info('Saving weights to ' + weights_file + '...')
            _log.info('Saving epoch logs to ' + log_file + '...')
            if isinstance(train, GeneratorType):
                # Training data is to be passed from generator
                self.model.fit_generator(generator=train,
                                    samples_per_epoch=nb_samples[Dataset.TRAIN.value],
                                    nb_epoch=max_epoch, 
                                    validation_data=validation_data,
                                    nb_val_samples=nb_samples[Dataset.VALID.value],
                                    callbacks=callbacks)
            else:
                # Training data is in numpy arrays
                self.model.fit(x=train.X_train, y=train.y_train, 
                          batch_size=batch_size, nb_epoch=max_epoch, shuffle=True, 
                          validation_data=validation_data,
                          callbacks=callbacks)
        except KeyboardInterrupt:
            # TODO let this filepath be set
            self._clean()

    def test(self, test, nb_samples=None):
        """
        Evaluates model's performance on test set
        """

        t_results = None
        if isinstance(test, GeneratorType):
            t_results = self.model.evaluate_generator(generator=test,
                                                val_samples=nb_samples)
        else:
            t_results = self.model.evaluate(test.X_test, test.y_test)

        _log.info(t_results)
        return t_results

    def predict(self, x, batch_size=32, verbose=1):
        """
        Generates model's predictions on unlabeled data

        # Arguments
            x: 
                unlabeled, one-dimensional Numpy array
            batch_size: 
                size of batches that are predicted upon at a time
            verbose:
                verbosity mode, 0 or 1
        """
        # if self.model is None:
        #     _log.exception('Model does not exist')
        #     logging.shutdown()

        _log.info('Predicting on input data...')
        y_predict = None
        if isinstance(x, GeneratorType):
            # Check if the following runs
            y_predict = self.model.predict_generator(generator=x,
                                                val_samples=nb_samples)
        else:
            y_predict = self.model.predict(x=x, batch_size=batch_size, verbose=verbose)

        _log.info(y_predict)
        return y_predict

        # TODO: move the notebook stuff into here
        # Plot ROC curves
        # plot_auc_curve(self.name, test.y_test[:, 0], y_predict[:, 0])
