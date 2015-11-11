# -*- coding: utf-8 -*-

import itertools as it
import numpy as np
import random
from .perceptron import Perceptron

class CombinatorialPerceptron(Perceptron):
    """Perceptron classifier with combinatorial feature explosion.

    This classifier assumes that feature vectors do not depend on the
    prediction, and combines each feature with each possible class label.
    Effectively, this means that perceptron weights are a matrix, with all
    possible features in one dimension, and all possible class labels in the
    other.

    Using this classifier with a feature extractor that returns features "A",
    "B", "C" and class labels "0" and "1" is equivalent to using a generative
    perceptron with a feature extractor that returns features "A&0", "A&1",
    "B&0", "B&1", "C&0", "C&1".  However, this implementation has a potentially
    huge performance benefit in this special case where each feature is combined
    with each class label.

    Use this classifier, for example, for the AND/OR problem or for POS tagging.
    """

    def predict_vector(self, vec):
        return np.argmax(np.dot(vec, self._w))

    def _predict_all_independent(self, x, as_label=True):
        assert all([isinstance(e, np.ndarray) for e in x])
        guesses = np.argmax(np.dot(x, self._w), axis=1).transpose()
        return self._label_mapper.get_names(guesses) if as_label else guesses

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

    def _train_common(self, x, y, seed=1):
        if self.sequenced:
            train_func = self._perform_train_iteration_sequenced
            eval_func = self._evaluate_training_set_sequenced
            preprocess_train = self._preprocess_train_sequenced
        else:
            train_func = self._perform_train_iteration_independent
            eval_func = self._evaluate_training_set_independent
            preprocess_train = self._preprocess_train_independent

        (x, y) = preprocess_train(x, y)
        self._w = np.zeros((self.feature_count, self.label_count))
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
            if self.averaged:
                all_w.append(self._w.copy())

        if self.averaged:
            if self.sequenced: # check if feature count changed between iterations
                for w in all_w:
                    if w.shape != (self.feature_count, self.label_count):
                        w.resize((self.feature_count, self.label_count))
            self._w = sum(all_w) / len(all_w)

    ############################################################################
    #### Standard (independent) prediction #####################################
    ############################################################################

    _train_independent = _train_common

    def _preprocess_labels_independent(self, labels):
        """Preprocess a full vector/list of class labels.

        Stores the number of unique class labels, and returns a numpy array of
        all labels.
        """
        self._label_mapper.reset()
        labels = np.array(self._label_mapper.map_list(labels))
        self.label_count = len(self._label_mapper)
        return labels

    def _preprocess_train_independent(self, x, y):
        assert len(x) == len(y)
        new_x = self._preprocess_data(x)
        new_y = self._preprocess_labels_independent(y)
        return (new_x, new_y)

    def _perform_train_iteration_independent(self, x, y, permutation):
        for n in range(len(x)):
            idx = permutation[n]
            guess = np.argmax(np.dot(x[idx], self._w)) # predict_vector
            if guess != y[idx]:
                # update step
                self._w[:, y[idx]] += self.learning_rate * x[idx]
                self._w[:, guess]  -= self.learning_rate * x[idx]

    def _evaluate_training_set_independent(self, x, y):
        correct = sum(self._predict_all_independent(x, as_label=False) == y)
        return 1.0 * correct / len(x)

    ############################################################################
    #### Sequenced prediction ##################################################
    ############################################################################

    _train_sequenced = _train_common

    def _preprocess_labels_sequenced(self, label_seq):
        self._label_mapper.reset()
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

    def _perform_train_iteration_sequenced(self, x, y, permutation):
        for n in range(len(x)):
            idx = permutation[n]
            (pad_x, history, start_pos) = self._initialize_sequence(x[idx])
            truth_seq = y[idx]

            # loop over sequence elements
            for pos in range(start_pos, start_pos + len(x[idx])):
                vec = self._feature_extractor.get_vector(
                    pad_x, pos, history=history
                    )
                if len(vec) > self._w.shape[0]:
                    self._w.resize((self.feature_count, self.label_count))
                guess = np.argmax(np.dot(vec, self._w)) # predict_vector
                truth = truth_seq[pos - self._left_context_size]
                if guess != truth:
                    # update step
                    self._w[:, truth] += self.learning_rate * vec
                    self._w[:, guess] -= self.learning_rate * vec
                history.append(self._label_mapper.get_name(guess))

    def _evaluate_training_set_sequenced(self, x, y):
        # TODO: could we skip this step and use the accuracy of the
        # prediction we already make during training? this is less accurate,
        # but potentially much faster on a huge dataset
        correct = 0
        total = 0
        for y_pred, y_truth in it.izip(self._predict_all_sequenced(x, as_label=False), y):
            correct += sum(y_pred == y_truth)
            total += len(y_pred)
        return 1.0 * correct / total
