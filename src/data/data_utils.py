import numpy as np
import h5py
import scipy.io
from math import floor
from enum import Enum

from collections import namedtuple as tuple

class Dataset(Enum):
    TRAIN=0
    VALID=1
    TEST=2

class DataFile(Enum):
    NAME=0
    X=1
    Y=2

class ModelData(object):

    def __init__(self,
                 shrink_size=(1.0, 1.0, 1.0),
                 batch_size=100,
                 train=('data/processed/train.mat', 'trainxdata', 'traindata'),
                 valid=('data/processed/valid.mat', 'validxdata', 'validdata'),
                 test=('data/processed/test.mat', 'testxdata', 'testdata')):

        self._shrink_size_=shrink_size
        self._batch_size_=batch_size
        self._train_=train
        self._valid_=valid
        self._test_=test

    def open_train_file(self, train=None):
        """
        Opens the file based off of the column names specified in 'train'

        # Arguments
            train:
                tuple (file name, x column, y column)
                to pull the x and y data from

        # Returns
            dict containing the contents of a h5py loaded file
        """
        if (train is not None):
            self._train_=train
        trainmat = h5py.File(self._train_[DataFile.NAME.value])
        return trainmat

    def open_valid_file(self, valid=None):
        """
        Opens the file based off of the column names specified in 'valid'

        # Arguments
            valid:
                tuple (file name, x column, y column)
                to pull the x and y data from

        # Returns
            dict containing the contents of a scipy loaded file
        """
        if (valid is not None):
            self._valid_=valid
        validmat = scipy.io.loadmat(self._valid_[DataFile.NAME.value])
        return validmat

    def open_test_file(self, test=None):
        """
        Opens the file based off of the column names specified in 'test'

        # Arguments
            test:
                tuple (file name, x column, y column)
                to pull the x and y data from

        # Returns
            dict containing the contents of a scipy loaded file
        """
        if (test is not None):
            self._test_=test
        testmat = scipy.io.loadmat(self._test_[DataFile.NAME.value])
        return testmat

    def set_shrink_size(self, train=1.0, valid=1.0, test=1.0):
        """
        Sets the float ratio of each truncated dataset to their respective full datasets.
        """
        self._shrink_size_=(train, valid, test)

    def set_nb_samples(self, train=None, valid=None, test=None):
        """
        Set the number of samples to use before moving to next epoch
        """
        if train is not None:
            self.nb_train_samples

    def set_batch_size(self, batch_size=None):
        """
        Sets the self._batch_size_
        """
        self._batch_size_=batch_size

    def _get_shrunk_array(self, f, ind, shrink_size=(1.0, 1.0, 1.0)):
        """
        The shrink_size must be a tuple of the same size as the array's shape
        For now, this is assumed to be 2D or 3D

        # Returns
            numpy array of reduced dataset size
        """
        dim = f[ind].shape
        if (f[ind].ndim == 2):
            return np.array(f[ind][:int(round(shrink_size[0] * dim[0])),
                                   :int(round(shrink_size[1] * dim[1]))])
        elif (f[ind].ndim == 3):
            return np.array(f[ind][:int(round(shrink_size[0] * dim[0])),
                                   :int(round(shrink_size[1] * dim[1])),
                                   :int(round(shrink_size[2] * dim[2]))])


    def get_data_tuples(self, shrink_size=(1.0, 1.0, 1.0)):
        """
        Returns truncated versions of test, validation, and train data.
        The shrink_size is a tuple of the ratio of 
        the truncated datasets to the full datasets.

        # Arguments
            shrink_size:
                tuple of the ratios (train, valid, test)
                of the truncated dataset to the full dataset

        # Returns
            named tuples ((X_train, y_train), 
                          (X_valid, y_valid), 
                          (X_test, y_test))
        """
        return (self.get_train_tuple(),
                self.get_valid_tuple(),
                self.get_test_tuple())

    def get_train_tuple(self, shrink_size=None):
        """
        Returns truncated version of the test data from a h5py file.
        The shrink_size is the ratio of 
        the truncated dataset to the full dataset.
        Passing a shrink_size will not modify the self._shrink_size_

        # Arguments
            shrink_size:
                float ratio of the truncated dataset to the full dataset

        # Returns
            named tuple (X_train, y_train)
        """
        if shrink_size is None:
            shrink_size = self._shrink_size_[Dataset.TRAIN.value]

        trainmat = self.open_train_file()

        # Reduce number of samples
        # H5py file is in (columns, rows, samples) and (classes, samples)
        X_train = np.transpose(
            self._get_shrunk_array(trainmat, self._train_[DataFile.X.value], (1, 1, shrink_size)),
            axes = (2, 0, 1))
        y_train = self._get_shrunk_array(trainmat, self._train_[DataFile.Y.value], (1, shrink_size)).T

        return tuple('train_tuple', 'X_train y_train')(X_train, y_train)

    def get_valid_tuple(self, shrink_size=None):
        """
        Returns truncated version of the validation data from a scipy.io file
        The shrink_size is the ratio of 
        the truncated dataset to the full dataset.
        Passing a shrink_size will not modify the self._shrink_size_

        # Arguments
            shrink_size:
                float ratio of the truncated dataset to the full dataset

        # Returns
            named tuple (X_valid, y_valid)
        """
        if shrink_size is None:
            shrink_size = self._shrink_size_[Dataset.VALID.value]

        validmat = self.open_valid_file()

        # Reduce number of samples
        # Scipy.io mat is in (samples, rows, columns) and (samples, classes)
        X_valid = np.transpose(
            self._get_shrunk_array(validmat, 
                                   self._valid_[DataFile.X.value], 
                                   (shrink_size, 1, 1)),
            axes = (0, 2, 1))
        y_valid = self._get_shrunk_array(validmat, 
                                         self._valid_[DataFile.Y.value], 
                                         (shrink_size, 1))

        return tuple('valid_tuple', 'X_valid y_valid')(X_valid, y_valid)

    def get_test_tuple(self, shrink_size=None):
        """
        Returns truncated version of the test data from a scipy.io file
        The shrink_size is the ratio of 
        the truncated dataset to the full dataset.
        Passing a shrink_size will not modify the self._shrink_size_

        # Arguments
            shrink_size:
                float ratio of the truncated dataset to the full dataset

        # Returns
            named tuple (X_test, y_test)
        """
        if shrink_size is None:
            shrink_size = self._shrink_size_[Dataset.TEST.value]

        testmat = self.open_test_file()

        # Reduce number of samples
        # Scipy.io mat is in (samples, rows, columns) and (samples, classes)
        X_test = np.transpose(
            self._get_shrunk_array(testmat, 
                                   self._test_[1], 
                                   (shrink_size, 1, 1)),
            axes = (0, 2, 1))
        y_test = self._get_shrunk_array(testmat, 
                                        self._test_[2], 
                                        (shrink_size, 1))

        return tuple('test_tuple', 'X_test y_test')(X_test, y_test)

    def get_data_tuples_generator(self, 
                                  shrink_size=None, 
                                  nb_samples=None, 
                                  batch_size=None):
        """
        Returns three generator that yield truncated versions
        of the training, validation, and test data.

        This function will not modify the class' member variables

        # Arguments
            shrink_size:
                tuple of the ratios (train, valid, test)
                of the truncated dataset to the full dataset
            nb_samples:
                tuple of the number of samples (train, valid, test)
                to use from each dataset
            batch_size:
                size of each mini batch during training
        """
        if shrink_size is None:
            shrink_size = self._shrink_size_
        if nb_samples is None:
            nb_samples = (None, None, None)
        if batch_size is None:
            batch_size = self._batch_size_

        return (self.get_train_tuple_generator(shrink_size=shrink_size[Dataset.TRAIN.value],
                                               nb_samples=nb_samples[Dataset.TRAIN.value],
                                               batch_size=batch_size),

                self.get_valid_tuple_generator(shrink_size=shrink_size[Dataset.VALID.value],
                                               nb_samples=nb_samples[Dataset.VALID.value],
                                               batch_size=batch_size),

                self.get_test_tuple_generator(shrink_size=shrink_size[Dataset.TEST.value],
                                              nb_samples=nb_samples[Dataset.TEST.value],
                                              batch_size=batch_size))

    def get_train_tuple_generator(self, 
                                  shrink_size=None, 
                                  nb_samples=None, 
                                  batch_size=None):
        """
        Creates a generator that yields a truncated version
        of the train data from a h5py file.

        This function will not modify the class' member variables

        # Arguments
            shrink_size:
                float ratio of the truncated dataset to the full dataset
            nb_samples:
                maximum number of samples; must be divisible by batch_size
            batch_size:
                size of each mini batch during training

        # Yields
            named tuple (X_train, y_train)
        """
        if shrink_size is None:
            shrink_size = self._shrink_size_[Dataset.TRAIN.value]
        if batch_size is None:
            batch_size = self._batch_size_

        train_tuple = self.get_train_tuple(shrink_size=shrink_size)

        max_batches = floor(train_tuple.X_train.shape[0] / batch_size)

        # Set the number of batches to either 
        # the maximum or the greatest possible
        if nb_samples is None or nb_samples >= (max_batches * batch_size):
            nb_batches = max_batches
        else:
            if nb_samples % batch_size != 0:
                sys.exit('ERROR: nb_samples is not divisible by batch_size')
            nb_batches = int(nb_samples / batch_size)
        # Yield the next batch
        while 1:
            for i in range (0, nb_batches):
                yield tuple('train_tuple', 'X_train y_train') \
                        (train_tuple.X_train[i * batch_size : (i+1) * batch_size],
                         train_tuple.y_train[i * batch_size : (i+1) * batch_size])

    def get_valid_tuple_generator(self, 
                                  shrink_size=None, 
                                  nb_samples=None, 
                                  batch_size=None):
        """
        Creates a generator that yields a truncated version
        of the validation data from a scipy.io file

        This function will not modify the class' member variables

        # Arguments
            shrink_size:
                float ratio of the truncated dataset to the full dataset
            nb_samples:
                maximum number of samples; must be divisible by batch_size
            batch_size:
                size of each mini batch during training

        # Yields
            named tuple (X_valid, y_valid)
        """
        if shrink_size is None:
            shrink_size = self._shrink_size_[Dataset.VALID.value]
        if batch_size is None:
            batch_size = self._batch_size_

        valid_tuple = self.get_valid_tuple(shrink_size=shrink_size)

        max_batches = floor(valid_tuple.X_valid.shape[0] / batch_size)

        # Set the number of batches to either 
        # the maximum or the greatest possible
        if nb_samples is None or nb_samples >= (max_batches * batch_size):
            nb_batches = max_batches
        else:
            if nb_samples % batch_size != 0:
                sys.exit('ERROR: nb_samples is not divisible by batch_size')
            nb_batches = int(nb_samples / batch_size)

        # Yield the next batch
        while 1:
            for i in range (0, nb_batches):
                yield tuple('valid_tuple', 'X_valid y_valid') \
                        (valid_tuple.X_valid[i * batch_size : (i+1) * batch_size],
                         valid_tuple.y_valid[i * batch_size : (i+1) * batch_size])

    def get_test_tuple_generator(self, 
                                 shrink_size=None, 
                                 nb_samples=None, 
                                 batch_size=None):
        """
        Creates a generator that yields a truncated version
        of the test data from a scipy.io file

        This function will not modify the class' member variables

        # Arguments
            shrink_size:
                float ratio of the truncated dataset to the full dataset
            nb_samples:
                maximum number of samples; must be divisible by batch_size
            batch_size:
                size of each mini batch during training

        # Yields
            named tuple (X_test, y_test)
        """
        if shrink_size is None:
            shrink_size = self._shrink_size_[Dataset.TEST.value]
        if batch_size is None:
            batch_size = self._batch_size_

        test_tuple = self.get_test_tuple(shrink_size=shrink_size)

        max_batches = floor(test_tuple.X_test.shape[0] / batch_size)

        # Set the number of batches to either 
        # the maximum or the greatest possible
        if nb_samples is None or nb_samples >= (max_batches * batch_size):
            nb_batches = max_batches
        else:
            if nb_samples % batch_size != 0:
                sys.exit('ERROR: nb_samples is not divisible by batch_size')
            nb_batches = int(nb_samples / batch_size)

        # Yield the next batch
        while 1:
            for i in range (0, nb_batches):
                yield tuple('test_tuple', 'X_test y_test') \
                        (test_tuple.X_test[i * batch_size : (i+1) * batch_size],
                         test_tuple.y_test[i * batch_size : (i+1) * batch_size])

    def nb_train_samples(self, shrink_size=None):
        """
        # Arguments
            shrink_size:
                float ratio of the truncated dataset to the full dataset

        # Returns
            number of training samples, possibly reduced by 
            the class's shrink size
        """
        if shrink_size is None:
            shrink_size = self._shrink_size_[Dataset.TRAIN.value]

        tuple = self.get_train_tuple(shrink_size=shrink_size)
        return tuple.X_train.shape[0]

    def nb_valid_samples(self, shrink_size=None):
        """
        # Arguments
            shrink_size:
                float ratio of the truncated dataset to the full dataset

        # Returns
            number of validation samples, possibly reduced by 
            the class's shrink size
        """
        if shrink_size is None:
            shrink_size = self._shrink_size_[Dataset.VALID.value]

        tuple = self.get_valid_tuple(shrink_size=shrink_size)
        return tuple.X_valid.shape[0]

    def nb_test_samples(self, shrink_size=None):
        """
        # Arguments
            shrink_size:
                float ratio of the truncated dataset to the full dataset

        # Returns
            number of test samples, possibly reduced by 
            the class's shrink size
        """
        if shrink_size is None:
            shrink_size = self._shrink_size_[Dataset.TEST.value]

        tuple = self.get_test_tuple(shrink_size=shrink_size)
        return tuple.X_test.shape[0]

    def get_batch_size(self):
        """
        # Returns
            size of each mini batch
        """
        return self._batch_size_
