# -*- coding: utf-8 -*-

import itertools as it
import numpy as np
from ..perceptron import Perceptron

class CombinatorialPerceptron_Numpy(Perceptron):
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

    @property
    def all_labels(self):
        try:
            return self._all_labels
        except AttributeError:
            self._all_labels = sorted(self._label_mapper)
            return self._all_labels

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
                    padded_x, i, history=history, grow=False
                    )
                guess = self.predict_vector(vector)
                history.append(self._label_mapper.get_name(guess))
                guesses = history[self._left_context_size:]
            return guesses
        else:
            guess = self.predict_vector(
                self._feature_extractor.get_vector(x, grow=False)
                )
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

    def print_weights(self):
        all_labels = self.all_labels
        sorted_indices = self._label_mapper.map_list(all_labels)
        # header
        print("\t" + "\t".join(all_labels).encode("utf-8"))
        # feature weights
        for (idx, label_weights) in enumerate(self._w):
            if self._feature_extractor is not None:
                # urgh
                feat = self._feature_extractor._label_mapper.get_name(idx)
            else:
                feat = unicode(idx)
            row = [feat]
            row.extend(label_weights[sorted_indices])
            print("\t".join(map(unicode, row)).encode("utf-8"))

    def prune_weights(self):
        prunable = []
        for (idx, label_weights) in enumerate(self._w):
            if all((abs(w) < self.prune_limit for w in label_weights)):
                prunable.append(idx)
        if prunable:
            self._w = np.delete(self._w, prunable, axis=0)
            if self._feature_extractor is not None:
                self._feature_extractor._label_mapper.prune_indices(prunable)

    ############################################################################
    #### Standard (independent) prediction #####################################
    ############################################################################

    def _perform_train_iteration_independent(self, x, y, permutation):
        correct, total = 0, len(x)
        for n in range(len(x)):
            if (n % 100) == 0: self._progress(n)
            idx = permutation[n]
            guess = np.argmax(np.dot(x[idx], self._w)) # predict_vector
            if guess != y[idx]:
                # update step
                self._w[:, y[idx]] += self.learning_rate * x[idx]
                self._w[:, guess]  -= self.learning_rate * x[idx]
            else:
                correct += 1
        return (correct, total)

    ############################################################################
    #### Sequenced prediction ##################################################
    ############################################################################

    def _perform_train_iteration_sequenced(self, x, y, permutation):
        correct, total = 0, 0
        for n in range(len(x)):
            idx = permutation[n]
            (pad_x, history, start_pos) = self._initialize_sequence(x[idx])
            truth_seq = y[idx]

            # loop over sequence elements
            for pos in range(start_pos, start_pos + len(x[idx])):
                total += 1
                if (total % 100) == 0: self._progress(total)
                vec = self._feature_extractor.get_vector(
                    pad_x, pos, history=history, grow=True
                    )
                if len(vec) > self._w.shape[0]:
                    self._w.resize((self.feature_count, len(self._label_mapper)))
                guess = np.argmax(np.dot(vec, self._w)) # predict_vector
                truth = truth_seq[pos - self._left_context_size]
                if guess != truth:
                    # update step
                    self._w[:, truth] += self.learning_rate * vec
                    self._w[:, guess] -= self.learning_rate * vec
                else:
                    correct += 1
                history.append(self._label_mapper.get_name(guess))
        return (correct, total)
