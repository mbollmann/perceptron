# -*- coding: utf-8 -*-

import itertools as it
import numpy as np
import random
from .label_mapper import LabelMapper

class Perceptron(object):
    """A perceptron classifier.
    """
    _label_mapper = None  # maps class labels to vector indices
    label_count = 0
    feature_count = 0

    # for sequence-based prediction:
    _left_context_template = "__BEGIN_{0}__"
    _right_context_template = "__END_{0}__"
    _initial_history_template = "__BEGIN_TAG_{0}__"
    _left_context_size = 0
    _left_context = []
    _right_context = []
    _initial_history = []

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
        return self._feature_extractor

    @feature_extractor.setter
    def feature_extractor(self, obj):
        self._feature_extractor = obj
        if obj is not None:
            self.set_context_attributes(obj)

    def predict_vector(self, vec):
        """Predict the class label of a given feature vector.
        """
        raise NotImplementedError("predict_vector function not implemented")

    def predict(self, x):
        """Predict the class label of a given data point.
        """
        guess = self.predict_vector(self._feature_extractor.get_vector(x))
        if self._label_mapper is not None:
            return self._label_mapper.get_name(guess)
        return guess

    def predict_all(self, x):
        """Predict the class labels of a given dataset (= list of feature vectors).
        """
        return [self.predict(y) for y in x]

    def predict_sequence(self, x):
        """Predict the class labels of a given sequence of data points.

        Requires a feature extractor to be given; the feature extractor can
        derive its features from the full sequence of data points and the
        previous predictions.
        """
        (padded_x, history, startpos) = self._initialize_sequence(x)
        for i in range(startpos, startpos + len(x)):
            guess = self.predict_vector(
                self._feature_extractor.get_vector_seq(
                    padded_x, i, history=history
                ))
            history.append(guess)
        guesses = history[self._left_context_size:]
        if self._label_mapper is not None:
            return self._label_mapper.get_names(guesses)
        return guesses

    def train(self, x, y, seed=1):
        """Train the perceptron.

        Parameters:
          x - A list of numpy feature vectors
          y - A list of correct class labels
        """
        raise NotImplementedError("train function not implemented")

    def set_context_attributes(self, obj):
        """Set context attributes from an object providing context size,
        typically the feature extractor.

        Required for sequence-based prediction only.
        """
        (left_context_size, right_context_size) = obj.context_size
        self._left_context, self._right_context, self._initial_history = [], [], []
        self._left_context_size = left_context_size
        for i in range(left_context_size):
            self._left_context.append(self._left_context_template.format(i))
            self._initial_history.append(self._initial_history_template.format(i))
        for j in range(right_context_size):
            self._right_context.append(self._right_context_template.format(i))

    def _initialize_sequence(self, seq):
        """Prepare a sequence of data points for sequence-based prediction.

        Pads the sequence with dummy context, if required, and prepares the
        prediction history.
        """
        padded_seq = self._left_context + seq + self._right_context
        history = self._initial_history
        startpos = self._left_context_size
        return (padded_seq, history, startpos)

    def _preprocess_data(self, data):
        """Preprocess a full list of training data.
        """
        if len(data) < 1:
            self.feature_count = 0
            return data
        if not isinstance(data[0], np.ndarray):
            self._feature_extractor.init(data)
            self.feature_count = self._feature_extractor.feature_count
            data = [self._feature_extractor.get_vector(x) for x in data]
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
            self._label_mapper = LabelMapper()
            labels = np.array(self._label_mapper.map_list(labels))
        self.label_count = np.unique(labels).shape[0]
        return labels

    def _preprocess_train(self, x, y):
        assert len(x) == len(y)
        new_x = self._preprocess_data(x)
        new_y = self._preprocess_labels(y)
        return (new_x, new_y)
