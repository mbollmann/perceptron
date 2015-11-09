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
        if self._label_mapper is not None and as_label:
            return self._label_mapper.get_names(guesses)
        return guesses

    def _train_independent(self, x, y, seed=1):
        (x, y) = self._preprocess_train(x, y)
        self._w = np.zeros((self.feature_count, self.label_count))
        all_w = []

        for iteration in range(self.iterations):
            # random permutation
            np.random.seed(seed)
            permutation = np.random.permutation(len(x))
            seed += 1

            # loop over examples
            for n in range(len(x)):
                idx = permutation[n]
                guess = np.argmax(np.dot(x[idx], self._w)) # predict_vector
                if guess != y[idx]:
                    # update step
                    self._w[:, y[idx]] += self.learning_rate * x[idx]
                    self._w[:, guess]  -= self.learning_rate * x[idx]

            # evaluate
            correct = sum(self._predict_all_independent(x, as_label=False) == y)
            accuracy = 1.0 * correct / len(x)
            self._log("Iteration {0:2}:  accuracy {1:.4f}".format(iteration, accuracy))
            if self.averaged:
                all_w.append(self._w.copy())

        if self.averaged:
            self._w = sum(all_w) / len(all_w)

    def _train_sequenced(self, x, y, seed=1):
        # TODO: check what we can refactor into a common function
        (x, y) = self._preprocess_train(x, y)
        self._w = np.zeros((self.feature_count, self.label_count))
        all_w = []

        for iteration in range(self.iterations):
            # random permutation
            np.random.seed(seed)
            permutation = np.random.permutation(len(x))
            seed += 1

            # loop over examples
            for n in range(len(x)):
                idx = permutation[n]
                (pad_x, history, start_pos) = self._initialize_sequence(x[idx])
                truth_seq = y[idx]

                # loop over sequence elements
                for pos in range(start_pos, start_pos + len(x[idx])):
                    vec = self._feature_extractor.get_vector(
                        pad_x, pos, history=history
                        )
                    if len(vec) > self.feature_count:
                        # TODO: dynamic resizing
                        pass
                    guess = np.argmax(np.dot(vec, self._w)) # predict_vector
                    truth = truth_seq[pos]
                    if guess != truth:
                        # update step
                        self._w[:, truth] += self.learning_rate * vec
                        self._w[:, guess] -= self.learning_rate * vec
                    history.append(guess)

            # evaluate

            # TODO: could we skip this step and use the accuracy of the
            # prediction we already make during training? this is less accurate,
            # but potentially much faster on a huge dataset
            correct = 0
            total = 0
            for y_pred, y_truth in it.izip(self._predict_all_sequenced(x, as_label=False), y):
                correct += sum(y_pred == y_truth)
                total += len(y_pred)
            accuracy = 1.0 * correct / total
            self._log("Iteration {0:2}:  accuracy {1:.4f}".format(iteration, accuracy))
            if self.averaged:
                all_w.append(self._w.copy())

        if self.averaged:
            # TODO: if feature count changed during training, we must also
            # (potentially) resize the all_w elements
            self._w = sum(all_w) / len(all_w)
