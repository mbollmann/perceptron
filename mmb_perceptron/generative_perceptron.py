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

    def reset_weights(self):
        self._w = np.zeros(self.feature_count)

    def _resize_weights(self, w):
        if w.shape != (self.feature_count,):
            w.resize(self.feature_count, refcheck=False)

    def _preprocess_train(self, x, y):
        assert len(x) == len(y)
        self._feature_extractor.init(x)
        self._label_mapper.reset()
        return (x, y)

    def predict_features(self, features):
        """Predicts the best feature vector from a list of feature vectors.
        """
        if self.feature_count > self._w.shape[0]:
            self._w.resize(self.feature_count)
        p = np.dot(features, self._w)
        p_best = np.argwhere(p == np.amax(p)).flatten()
        if len(p_best) > 1:
            # there is more than one solution -- pick one randomly, since
            # the order returned by the feature extractor might be
            # deterministic
            return np.random.choice(p_best)
        else:
            return p_best[0]

    def predict(self, x):
        if self.sequenced:
            (padded_x, history, startpos) = self._initialize_sequence(x)
            for i in range(startpos, startpos + len(x)):
                (features, labels) = self._feature_extractor.generate_vector(
                    padded_x, i, history=history
                    )
                if self.feature_count > self._w.shape[0]:
                    self._w.resize(self.feature_count)
                guess = np.argmax(np.dot(features, self._w))
                history.append(labels[guess])
            guesses = history[self._left_context_size:]
            return guesses
        else:
            (features, labels) = self._feature_extractor.generate_vector(x)
            if self.feature_count > self._w.shape[0]:
                self._w.resize(self.feature_count)
            guess = np.argmax(np.dot(features, self._w))
            return labels[guess]

    ############################################################################
    #### Standard (independent) prediction #####################################
    ############################################################################

    def _perform_train_iteration_independent(self, x, y, permutation):
        for n in range(len(x)):
            idx = permutation[n]
            (features, _) = \
                self._feature_extractor.generate_vector(x[idx], truth=y[idx])
            guess = self.predict_features(features)
            if guess != 0:
                # update step
                self._w += self.learning_rate * features[0]
                self._w -= self.learning_rate * features[guess]

    def _evaluate_training_set_independent(self, x, y):
        correct = sum(a == b for (a, b) in \
                      it.izip(self.predict_all(x), y))
        return 1.0 * correct / len(x)

    ############################################################################
    #### Sequenced prediction ##################################################
    ############################################################################

    def _perform_train_iteration_sequenced(self, x, y, permutation):
        for n in range(len(x)):
            idx = permutation[n]
            (pad_x, history, start_pos) = self._initialize_sequence(x[idx])
            truth_seq = y[idx]

            # loop over sequence elements
            for pos in range(start_pos, start_pos + len(x[idx])):
                (features, labels) = self._feature_extractor.generate_vector(
                    pad_x, pos, history=history,
                    truth=truth_seq[pos - self._left_context_size]
                    )
                guess = self.predict_features(features)
                if guess != 0:
                    # update step
                    self._w += self.learning_rate * features[0]
                    self._w -= self.learning_rate * features[guess]
                history.append(labels[guess])

    def _evaluate_training_set_sequenced(self, x, y):
        # TODO: could we skip this step and use the accuracy of the
        # prediction we already make during training? this is less accurate,
        # but potentially much faster on a huge dataset
        correct = 0
        total = 0
        for y_pred, y_truth in it.izip(self.predict_all(x), y):
            correct += sum(a == b for (a, b) in it.izip(y_pred, y_truth))
            total += len(y_pred)
        return 1.0 * correct / total
