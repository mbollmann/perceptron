# -*- coding: utf-8 -*-

import itertools as it
import numpy as np
import random
from .feature_mapper import FeatureMapper

class Perceptron(object):
    """A perceptron classifier.
    """
    _fm = None
    _label_mapper = None
    label_count = 0
    feature_count = 0

    def __init__(self, iterations=5, learning_rate=1, averaging=True, \
                 feature_extractor=None, log_to=None):
        self.averaging = averaging
        self.feature_extractor = feature_extractor
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.log_to = log_to

    def _log(self, text):
        if self.log_to is not None:
            self.log_to.write(text)
            self.log_to.write("\n")

    @property
    def feature_extractor(self):
        return self._fe

    @feature_extractor.setter
    def feature_extractor(self, obj):
        self._fe = obj
        self._fm = FeatureMapper() if obj is not None else None

    def predict_vector(self, vec):
        """Predict the class label of a given feature vector.
        """
        raise NotImplementedError("predict_vector function not implemented")

    def predict_datapoint(self, x):
        """Predict the class label of a given data point.
        """
        guess = self.predict_vector(self._fm.map_to_vector(self._fe.get(x)))
        if self._label_mapper is not None:
            return self._label_mapper.get_name(guess)
        return guess

    def predict(self, x):
        """Predict the class label of a given dataset (= list of feature vectors).
        """
        raise NotImplementedError("predict function not implemented")

    def train(self, x, y, seed=1):
        """Train the perceptron.

        Parameters:
          x - A list of numpy feature vectors (or a matrix? idk yet)
          y - A list of correct class labels
        """
        raise NotImplementedError("train function not implemented")

    def _preprocess_data(self, data):
        """Preprocess a full list of training data.

        TODO: Currently only accepts lists of numpy vectors.
        """
        if len(data) < 1:
            self.feature_count = 0
            return data
        if not isinstance(data[0], np.ndarray):
            self._fe.init(data)
            self._fm.extend(self._fe.features)
            self.feature_count = self._fe.feature_count
            data = [self._fm.map_to_vector(self._fe.get(x)) for x in data]
        else:
            self.feature_count = data[0].shape[0]
        assert all(x.shape[0] == self.feature_count for x in data)
        return data

    def _preprocess_labels(self, labels):
        """Preprocess a full vector/list of class labels.

        Stores the number of unique class labels, and returns a numpy array of
        all labels.  If necessary, a feature mapper is instantiated for the
        conversion.
        """
        if not isinstance(labels, np.ndarray):
            self._label_mapper = FeatureMapper()
            labels = np.array(self._label_mapper.map_list(labels))
        self.label_count = np.unique(labels).shape[0]
        return labels

    def _preprocess_train(self, x, y):
        assert len(x) == len(y)
        new_x = self._preprocess_data(x)
        new_y = self._preprocess_labels(y)
        return (new_x, new_y)
