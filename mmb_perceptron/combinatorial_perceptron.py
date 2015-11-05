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

    def predict_all(self, x):
        assert all([isinstance(e, np.ndarray) for e in x])
        return np.argmax(np.dot(x, self._w), axis=1).transpose()

    def train(self, x, y, seed=1):
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
            correct = sum(self.predict_all(x) == y)
            accuracy = 1.0 * correct / len(x)
            self._log("Iteration {0:2}:  accuracy {1:.4f}".format(iteration, accuracy))
            if self.averaging:
                all_w.append(self._w.copy())

        if self.averaging:
            self._w = sum(all_w) / len(all_w)
