# -*- coding: utf-8 -*-

import itertools as it
import numpy as np
from ..perceptron import Perceptron

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

    def reset_weights(self):
        self._w = np.zeros((self.feature_count, len(self._label_mapper)))

    def _resize_weights(self, w):
        if w.shape != (self.feature_count, len(self._label_mapper)):
            w.resize((self.feature_count, len(self._label_mapper)), refcheck=False)

    def predict_vector(self, vec):
        return np.argmax(np.dot(vec, self._w))

    def predict(self, x):
        """Predict the class label of a given data point.
        """
        if self.sequenced:
            (padded_x, history, startpos) = self._initialize_sequence(x)
            for i in range(startpos, startpos + len(x)):
                vector = self._feature_extractor.get_vector(
                    padded_x, i, history=history
                    )
                if self.feature_count > self._w.shape[0]:
                    self._w.resize((self.feature_count, len(self._label_mapper)))
                guess = self.predict_vector(vector)
                history.append(self._label_mapper.get_name(guess))
                guesses = history[self._left_context_size:]
            return guesses
        else:
            guess = self.predict_vector(self._feature_extractor.get_vector(x))
            return self._label_mapper.get_name(guess)

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

    def _preprocess_train(self, x, y):
        assert len(x) == len(y)
        self._label_mapper.reset()
        if self.sequenced:
            # cannot preprocess the data (since vectors can depend on previous
            # predictions) except for forwarding it to the feature extractor
            self._feature_extractor.init(x)
            new_x = x
            new_y = [np.array(self._label_mapper.map_list(l)) for l in y]
        else:
            new_x = self._preprocess_data(x)
            new_y = np.array(self._label_mapper.map_list(y))
        return (new_x, new_y)

    ############################################################################
    #### Standard (independent) prediction #####################################
    ############################################################################

    def _perform_train_iteration_independent(self, x, y, permutation):
        for n in range(len(x)):
            idx = permutation[n]
            guess = np.argmax(np.dot(x[idx], self._w)) # predict_vector
            if guess != y[idx]:
                # update step
                self._w[:, y[idx]] += self.learning_rate * x[idx]
                self._w[:, guess]  -= self.learning_rate * x[idx]

    def _evaluate_training_set_independent(self, x, y):
        # more efficient than using _predict_all_independent
        guesses = np.argmax(np.dot(x, self._w), axis=1).transpose()
        correct = sum(guesses == y)
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
                vec = self._feature_extractor.get_vector(
                    pad_x, pos, history=history
                    )
                if len(vec) > self._w.shape[0]:
                    self._w.resize((self.feature_count, len(self._label_mapper)))
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
        for y_pred, y_truth in it.izip(self.predict_all(x), y):
            correct += sum(self._label_mapper.map_list(y_pred) == y_truth)
            total += len(y_pred)
        return 1.0 * correct / total
