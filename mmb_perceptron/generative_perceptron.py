# -*- coding: utf-8 -*-

import itertools as it
import numpy as np
import random
from .perceptron import Perceptron

class GenerativePerceptron(Perceptron):
    """Perceptron classifier with generation function.

    This classifier uses a generation function to get a list of prediction
    candidates for each data point.  These are then fed into the feature
    extractor, which is expected to generate a feature vector based on both the
    data point and the candidate prediction.  Perceptron weights are a vector,
    which is multiplied with the feature vector to obtain a score, and the best
    score from the generated feature vectors determines the prediction.
    """

    def _preprocess_train(self, x, y):
        assert len(x) == len(y)
        self._feature_extractor.init(x)
        self._label_mapper.reset()
        return (x, y)

    def _train_common(self, x, y, seed=1):
        if self.sequenced:
            train_func = self._perform_train_iteration_sequenced
            eval_func = self._evaluate_training_set_sequenced
        else:
            train_func = self._perform_train_iteration_independent
            eval_func = self._evaluate_training_set_independent

        (x, y) = self._preprocess_train(x, y)
        self._w = np.zeros(self.feature_count)
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
                    if w.shape != (self.feature_count,):
                        w.resize(self.feature_count)
            self._w = sum(all_w) / len(all_w)

    ############################################################################
    #### Standard (independent) prediction #####################################
    ############################################################################

    _train_independent = _train_common

    def _predict_independent(self, x, as_label=True):
        (features, labels) = self._feature_extractor.generate_vector(x)
        guess = np.argmax(np.dot(features, self._w))
        return labels[guess] if as_label else self._label_mapper[labels[guess]]

    def _perform_train_iteration_independent(self, x, y, permutation):
        for n in range(len(x)):
            idx = permutation[n]
            (features, _) = \
                self._feature_extractor.generate_vector(x[idx], truth=y[idx])
            if self.feature_count > self._w.shape[0]:
                self._w.resize(self.feature_count)
            p = np.dot(features, self._w)
            p_best = np.argwhere(p == np.amax(p)).flatten()
            if len(p_best) > 1:
                # there is more than one solution -- pick one randomly, since
                # the order returned by the feature extractor might be
                # deterministic
                guess = np.random.choice(p_best)
            else:
                guess = p_best[0]
            if guess != 0:
                # update step
                self._w += self.learning_rate * features[0]
                self._w -= self.learning_rate * features[guess]

    def _evaluate_training_set_independent(self, x, y):
        correct = sum(a == b for (a, b) in \
                      it.izip(self._predict_all_independent(x, as_label=True), y))
        return 1.0 * correct / len(x)
