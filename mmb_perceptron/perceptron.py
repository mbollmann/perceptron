# -*- coding: utf-8 -*-

import itertools as it
import numpy as np
from .label_mapper import LabelMapper

class Perceptron(object):
    """A perceptron classifier.

    This class implements methods common to all perceptron variants, but cannot
    be used by itself.  Always use derived classes instead.
    """
    _feature_extractor = None
    _w = None

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
        self._label_mapper = LabelMapper()
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
            if self.sequenced:
                self._set_context_attributes(obj)
            self._feature_extractor.sequenced = self.sequenced

    @property
    def sequenced(self):
        return self._sequenced

    @sequenced.setter
    def sequenced(self, status):
        self._sequenced = status
        if self._feature_extractor is not None:
            self._feature_extractor.sequenced = status

    def train(self, x, y, seed=1):
        """Train the perceptron on independent data points.

        Parameters:
          x - A list of data points or feature vectors, or (if
              sequenced) a list of data point/feature vector sequences
          y - A list of correct class labels
        """
        if self.sequenced:
            train_func = self._perform_train_iteration_sequenced
            eval_func = self._evaluate_training_set_sequenced
        else:
            train_func = self._perform_train_iteration_independent
            eval_func = self._evaluate_training_set_independent

        (x, y) = self._preprocess_train(x, y)
        self.reset_weights()
        all_w = []

        for iteration in range(self.iterations):
            # random permutation
            np.random.seed(seed)
            permutation = np.random.permutation(len(x))
            seed += 1

            # training
            train_func(x, y, permutation)

            # evaluation
            accuracy = eval_func(x, y)
            self._log("Iteration {0:2}:  accuracy {1:.4f}".format(iteration, accuracy))
            if self.averaged and self.iterations > 1:
                all_w.append(self._w.copy())

        if self.averaged and self.iterations > 1:
            self._log("Averaging weights...")
            self._w = self.average_weights(all_w)

    def average_weights(self, all_w):
        if self.sequenced: # check if feature count changed between iterations
            for w in all_w:
                self._resize_weights(w)
        return sum(all_w) / len(all_w)

    def predict_all(self, x):
        """Predict the class labels of a given dataset (= list of data points/sequences).

        The prediction function itself must be implemented by derived classes.
        """
        return [self.predict(y) for y in x]

    ############################################################################
    #### Serialization via pickle ##############################################
    ############################################################################

    def __getstate__(self):
        return {
            'averaged': self.averaged,
            'sequenced': self.sequenced,
            'feature_extractor': self._feature_extractor,
            'iterations': self.iterations,
            'label_mapper': self._label_mapper,
            'learning_rate': self.learning_rate,
            'weights': self._w
            }

    def __setstate__(self, state):
        self.averaged = state['averaged']
        self.sequenced = state['sequenced']
        self.feature_extractor = state['feature_extractor']
        self.iterations = state['iterations']
        self._label_mapper = state['label_mapper']
        self.learning_rate = state['learning_rate']
        self._w = state['weights']

    ############################################################################
    #### Functions to be implemented by derived classes ########################
    ############################################################################

    def predict(self, x):
        """Predict the class label of a given data point or sequence.
        """
        raise NotImplementedError("predictor functionality not implemented")

    def reset_weights(self):
        """Reset learned weights.
        """
        raise NotImplementedError("reset_weights function not implemented")

    def _resize_weights(self, w):
        """Resize weights dynamically, if needed.
        """
        raise NotImplementedError("_resize_weights function not implemented")

    def _perform_train_iteration_independent(self, x, y, permutation):
        raise NotImplementedError("training functionality not implemented")

    def _perform_train_iteration_sequenced(self, x, y, permutation):
        raise NotImplementedError("training functionality not implemented")

    def _evaluate_training_set_independent(self, x, y):
        raise NotImplementedError("training functionality not implemented")

    def _evaluate_training_set_sequenced(self, x, y):
        raise NotImplementedError("training functionality not implemented")

    ############################################################################
    #### Helper functions for sequenced prediction #############################
    ############################################################################

    def _set_context_attributes(self, obj):
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
        history = self._initial_history[:]
        startpos = self._left_context_size
        return (padded_seq, history, startpos)
