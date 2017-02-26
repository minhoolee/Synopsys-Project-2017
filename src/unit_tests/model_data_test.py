from src.data.data_utils import ModelData
import unittest
import h5py
import scipy.io
import numpy as np

from math import floor

# Constants based off of 'data/processed/{train.mat, valid.mat, test.mat}'
TRAIN_SAMPLES=4400000
VALID_SAMPLES=8000
TEST_SAMPLES=455024
WIDTH=4
LENGTH=1000
CLASSES=919
TRAIN_SHRINK=0.01
VALID_SHRINK=0.1
TEST_SHRINK=0.1
BATCH_SIZE=100

class ModelDataTest(unittest.TestCase):
    def test_shrink_size(self):
        data = ModelData()
        data.set_shrink_size(train=TRAIN_SHRINK, valid=VALID_SHRINK, test=TEST_SHRINK)
        train = data.get_train_tuple()
        self.assertEqual(train.X_train.shape, (TRAIN_SHRINK*TRAIN_SAMPLES, LENGTH, WIDTH))

    def test_open_file(self):
        data = ModelData()
        data.set_shrink_size(train=1, valid=1, test=1)
        train = data.open_train_file()
        valid = data.open_valid_file()
        test = data.open_test_file()
        train_ = h5py.File('data/processed/train.mat')
        valid_ = scipy.io.loadmat('data/processed/valid.mat')
        test_ = scipy.io.loadmat('data/processed/test.mat')
        self.assertEqual(train.keys(), train_.keys())
        self.assertEqual(valid.keys(), valid_.keys())
        self.assertEqual(test.keys(), test_.keys())

    def test_data_tuples(self):
        data = ModelData()
        train, valid, test = data.get_data_tuples()
        self.assertEqual(train.X_train.shape, (TRAIN_SAMPLES, LENGTH, WIDTH))
        self.assertEqual(train.y_train.shape, (TRAIN_SAMPLES, CLASSES))
        self.assertEqual(valid.X_valid.shape, (VALID_SAMPLES, LENGTH, WIDTH))
        self.assertEqual(valid.y_valid.shape, (VALID_SAMPLES, CLASSES))
        self.assertEqual(test.X_test.shape, (TEST_SAMPLES, LENGTH, WIDTH))
        self.assertEqual(test.y_test.shape, (TEST_SAMPLES, CLASSES))

    def test_individual_tuples(self):
        data = ModelData()
        train = data.get_train_tuple()
        valid = data.get_valid_tuple()
        test = data.get_test_tuple()
        self.assertEqual(train.X_train.shape, (TRAIN_SAMPLES, LENGTH, WIDTH))
        self.assertEqual(train.y_train.shape, (TRAIN_SAMPLES, CLASSES))
        self.assertEqual(valid.X_valid.shape, (VALID_SAMPLES, LENGTH, WIDTH))
        self.assertEqual(valid.y_valid.shape, (VALID_SAMPLES, CLASSES))
        self.assertEqual(test.X_test.shape, (TEST_SAMPLES, LENGTH, WIDTH))
        self.assertEqual(test.y_test.shape, (TEST_SAMPLES, CLASSES))

    def test_generator_wraparound(self):
        data = ModelData()
        nb_samples = 3 * BATCH_SIZE
        train, valid, test = data.get_data_tuples_generator(nb_samples=(nb_samples, nb_samples, nb_samples))

        # First batch
        train_batch_1 = next(train)
        valid_batch_1 = next(valid)
        test_batch_1 = next(test)

        # Second and third batches
        for i in range(2):
            self.check_generator_length(train=train, valid=valid, test=test)

        # Fourth batch should wrap back to first batch
        train_batch_4 = next(train)
        valid_batch_4 = next(valid)
        test_batch_4 = next(test)
        self.assertTrue(np.array_equal(train_batch_1.X_train, train_batch_4.X_train))
        self.assertTrue(np.array_equal(train_batch_1.y_train, train_batch_4.y_train))
        self.assertTrue(np.array_equal(valid_batch_1.X_valid, valid_batch_4.X_valid))
        self.assertTrue(np.array_equal(valid_batch_1.y_valid, valid_batch_4.y_valid))
        self.assertTrue(np.array_equal(test_batch_1.X_test, test_batch_4.X_test))
        self.assertTrue(np.array_equal(test_batch_1.y_test, test_batch_4.y_test))

    def check_generator_length(self, train=None, valid=None, test=None):
        train_batch = next(train)
        valid_batch = next(valid)
        test_batch = next(test)

        self.assertEqual(train_batch.X_train.shape, (BATCH_SIZE, LENGTH, WIDTH))
        self.assertEqual(train_batch.y_train.shape, (BATCH_SIZE, CLASSES))
        self.assertEqual(valid_batch.X_valid.shape, (BATCH_SIZE, LENGTH, WIDTH))
        self.assertEqual(valid_batch.y_valid.shape, (BATCH_SIZE, CLASSES))
        self.assertEqual(test_batch.X_test.shape, (BATCH_SIZE, LENGTH, WIDTH))
        self.assertEqual(test_batch.y_test.shape, (BATCH_SIZE, CLASSES))

    def test_generator_full(self):
        data = ModelData()
        train, valid, test = data.get_data_tuples_generator()

        # First batch
        train_batch_1 = next(train)
        valid_batch_1 = next(valid)
        test_batch_1 = next(test)

        # All batches until end
        self.check_train_generator_length(train)
        self.check_valid_generator_length(valid)
        self.check_test_generator_length(test)

        # Next batch should wrap back to first batch
        train_batch_begin = next(train)
        valid_batch_begin = next(valid)
        test_batch_begin = next(test)
        self.assertTrue(np.array_equal(train_batch_1.X_train, train_batch_begin.X_train))
        self.assertTrue(np.array_equal(train_batch_1.y_train, train_batch_begin.y_train))
        self.assertTrue(np.array_equal(valid_batch_1.X_valid, valid_batch_begin.X_valid))
        self.assertTrue(np.array_equal(valid_batch_1.y_valid, valid_batch_begin.y_valid))
        self.assertTrue(np.array_equal(test_batch_1.X_test, test_batch_begin.X_test))
        self.assertTrue(np.array_equal(test_batch_1.y_test, test_batch_begin.y_test))

    def check_train_generator_length(self, train):
        for i in range(1, TRAIN_SAMPLES // BATCH_SIZE):
            train_batch = next(train)
            self.assertEqual(train_batch.X_train.shape, (BATCH_SIZE, LENGTH, WIDTH))
            self.assertEqual(train_batch.y_train.shape, (BATCH_SIZE, CLASSES))

    def check_valid_generator_length(self, valid):
        for i in range(1, VALID_SAMPLES // BATCH_SIZE):
            valid_batch = next(valid)
            self.assertEqual(valid_batch.X_valid.shape, (BATCH_SIZE, LENGTH, WIDTH))
            self.assertEqual(valid_batch.y_valid.shape, (BATCH_SIZE, CLASSES))

    def check_test_generator_length(self, test):
        for i in range(1, TEST_SAMPLES // BATCH_SIZE):
            test_batch = next(test)
            self.assertEqual(test_batch.X_test.shape, (BATCH_SIZE, LENGTH, WIDTH))
            self.assertEqual(test_batch.y_test.shape, (BATCH_SIZE, CLASSES))

    def test_setters_getters(self):
        data = ModelData()
        data.set_shrink_size(train=TRAIN_SHRINK, valid=VALID_SHRINK, test=TEST_SHRINK)
        self.assertEqual(data.nb_train_samples(), floor(TRAIN_SHRINK * TRAIN_SAMPLES))
        self.assertEqual(data.nb_valid_samples(), floor(VALID_SHRINK * VALID_SAMPLES))
        self.assertEqual(data.nb_test_samples(), floor(TEST_SHRINK * TEST_SAMPLES))

if __name__ == '__main__':
    unittest.main()
