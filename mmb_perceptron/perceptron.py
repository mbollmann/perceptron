# -*- coding: utf-8 -*-

import itertools as it
import numpy as np
import random
from .label_mapper import LabelMapper

class Perceptron(object):
    """A perceptron classifier.
    """
    _feature_extractor = None
    _label_mapper = None  # maps class labels to vector indices
    label_count = 0

    # for sequence-based prediction:
    _left_context_template = "__BEGIN_{0}__"
    _right_context_template = "__END_{0}__"
    _initial_history_template = "__BEGIN_TAG_{0}__"
    _left_context_size = 0
    _left_context = []
    _right_context = []
    _initial_history = []

    def __init__(self, iterations=5, learning_rate=1, averaged=True, \
                 sequenced=False, feature_extractor=None, log_to=None):
        self.averaged = averaged
        self.sequenced = sequenced
        self.feature_extractor = feature_extractor
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.log_to = log_to

    def _log(self, text):
        if self.log_to is not None:
            self.log_to.write(text)
            self.log_to.write("\n")

    @property
    def feature_count(self):
        if self._feature_extractor is not None:
            return self._feature_extractor.feature_count
        return self._feature_count

    @feature_count.setter
    def feature_count(self, value):
        if self._feature_extractor is not None:
            raise AttributeError(("cannot set feature_count manually "
                                  "when using a feature_extractor"))
        self._feature_count = value

    @property
    def feature_extractor(self):
        return self._feature_extractor

    @feature_extractor.setter
    def feature_extractor(self, obj):
        self._feature_extractor = obj
        if obj is not None:
            self.set_context_attributes(obj)
            self._feature_extractor.sequenced = self.sequenced

    @property
    def sequenced(self):
        return self._sequenced

    @sequenced.setter
    def sequenced(self, status):
        self._sequenced = status
        if self._feature_extractor is not None:
            self._feature_extractor.sequenced = status
        if status:
            self.predict = self._predict_sequenced
            self.predict_all = self._predict_all_sequenced
            self.train = self._train_sequenced
            self._preprocess_train = self._preprocess_train_sequenced
            self.set_context_attributes = self._set_context_attributes_sequenced
        else:
            self.predict = self._predict_independent
            self.predict_all = self._predict_all_independent
            self.train = self._train_independent
            self._preprocess_train = self._preprocess_train_independent
            self.set_context_attributes = self._set_context_attributes_independent

    def predict_vector(self, vec):
        """Predict the class label of a given feature vector.
        """
        raise NotImplementedError("predict_vector function not implemented")

    ############################################################################
    #### PRIVATE METHODS #######################################################
    ############################################################################

    def _preprocess_data(self, data):
        """Preprocess a full list of training data.
        """
        if len(data) < 1:
            self.feature_count = 0
            return data
        if not isinstance(data[0], np.ndarray):
            self._feature_extractor.init(data)
            data = [self._feature_extractor.get_vector(x) for x in data]
        else:
            self.feature_count = data[0].shape[0]
        if not all(x.shape[0] == self.feature_count for x in data):
            raise ValueError("error converting data")
        return data

    ############################################################################
    #### Standard (independent) prediction #####################################
    ############################################################################

    def _predict_independent(self, x, as_label=True):
        """Predict the class label of a given data point.
        """
        guess = self.predict_vector(self._feature_extractor.get_vector(x))
        if self._label_mapper is not None and as_label:
            return self._label_mapper.get_name(guess)
        return guess

    def _predict_all_independent(self, x, as_label=True):
        """Predict the class labels of a given dataset (= list of feature vectors).
        """
        return [self._predict_independent(y, as_label=as_label) for y in x]

    def _train_independent(self, x, y, seed=1):
        """Train the perceptron on independent data points.

        Parameters:
          x - A list of data points or feature vectors
          y - A list of correct class labels
        """
        raise NotImplementedError("train function not implemented")

    def _set_context_attributes_independent(self, _):
        pass

    def _preprocess_labels_independent(self, labels):
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

    def _preprocess_train_independent(self, x, y):
        assert len(x) == len(y)
        new_x = self._preprocess_data(x)
        new_y = self._preprocess_labels_independent(y)
        return (new_x, new_y)

    ############################################################################
    #### Sequenced prediction ##################################################
    ############################################################################

    def _predict_sequenced(self, x, as_label=True):
        """Predict the class labels of a given sequence of data points.

        Requires a feature extractor to be given; the feature extractor can
        derive its features from the full sequence of data points and the
        previous predictions.
        """
        (padded_x, history, startpos) = self._initialize_sequence(x)
        for i in range(startpos, startpos + len(x)):
            guess = self.predict_vector(
                self._feature_extractor.get_vector(
                    padded_x, i, history=history
                ))
            if self._label_mapper is not None:
                history.append(self._label_mapper.get_name(guess))
            else:
                history.append(guess)
        guesses = history[self._left_context_size:]
        if self._label_mapper is not None and not as_label:
            return self._label_mapper.map_list(guesses)
        return guesses

    def _predict_all_sequenced(self, x, as_label=True):
        """Predict the class labels of a given sequential dataset.
        """
        return [self._predict_sequenced(y, as_label=as_label) for y in x]

    def _train_sequenced(self, x, y, seed=1):
        """Train the perceptron on fixed sequences of data points.

        Parameters:
          x - A list of sequences of data points
          y - A list of sequences of correct class labels
        """
        raise NotImplementedError("train_seq function not implemented")

    def _set_context_attributes_sequenced(self, obj):
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

    def _preprocess_labels_sequenced(self, label_seq):
        self._label_mapper = LabelMapper()
        new_seq = [np.array(self._label_mapper.map_list(l)) for l in label_seq]
        self.label_count = len(self._label_mapper)
        return new_seq

    def _preprocess_train_sequenced(self, x, y):
        assert len(x) == len(y)
        # cannot preprocess the data (since vectors can depend on previous
        # predictions) except for forwarding it to the feature extractor
        self._feature_extractor.init(x)
        new_y = self._preprocess_labels_sequenced(y)
        return (x, new_y)

    def _initialize_sequence(self, seq):
        """Prepare a sequence of data points for sequence-based prediction.

        Pads the sequence with dummy context, if required, and prepares the
        prediction history.
        """
        padded_seq = self._left_context + seq + self._right_context
        history = self._initial_history[:]
        startpos = self._left_context_size
        return (padded_seq, history, startpos)
