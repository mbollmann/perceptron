# -*- coding: utf-8 -*-

import itertools as it
import numpy as np
import operator as op
from ..perceptron import Perceptron

class defaultdict_numpy(dict):
    def __init__(self, dim):
        self._dim = dim

    def __missing__(self, key):
        self[key] = np.zeros(self._dim)
        return self[key]


class CombinatorialPerceptron_Mixed(Perceptron):
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
        self._w = defaultdict_numpy(len(self._label_mapper))

    def _resize_weights(self, w):
        pass

    def predict_features(self, features):
        scores = sum(self._w.get(f, 0) * v for f, v in features.iteritems())
        return self._label_mapper.get_name(np.argmax(scores))

    def predict(self, x):
        """Predict the class label of a given data point.
        """
        if self.sequenced:
            (padded_x, history, startpos) = self._initialize_sequence(x)
            for i in range(startpos, startpos + len(x)):
                features = self._feature_extractor.get(
                    padded_x, i, history=history
                    )
                guess = self.predict_features(features)
                history.append(guess)
            guesses = history[self._left_context_size:]
            return guesses
        else:
            guess = self.predict_features(self._feature_extractor.get(x))
            return guess

    def _preprocess_data(self, data):
        """Preprocess a full list of training data.
        """
        if len(data) < 1:
            self.feature_count = 0
            return data
        self._feature_extractor.init(data)
        data = [self._feature_extractor.get(x) for x in data]
        return data

    def _preprocess_train(self, x, y):
        assert len(x) == len(y)
        self._label_mapper.reset()
        if self.sequenced:
            # cannot preprocess the data (since vectors can depend on previous
            # predictions) except for forwarding it to the feature extractor
            self._feature_extractor.init(x)
            new_x = x
            for seq_y in y:
                self._label_mapper.extend(seq_y)
        else:
            new_x = self._preprocess_data(x)
            self._label_mapper.extend(y)
        return (new_x, y)

    def average_weights(self, all_w):
        averaged = defaultdict_numpy(len(self._label_mapper))
        divisor = float(len(all_w))
        final_w = all_w[-1]
        for feat, label_weights in final_w.iteritems():
            averaged[feat] = sum((_w[feat] for _w in all_w)) / divisor
        return averaged

    def print_weights(self):
        all_labels = self.all_labels
        sorted_indices = self._label_mapper.map_list(all_labels)
        # header
        print("\t" + "\t".join(all_labels).encode("utf-8"))
        # feature weights
        for feature, label_weights in self._w.iteritems():
            row = [feature]
            row.extend(label_weights[sorted_indices])
            print("\t".join(map(unicode, row)).encode("utf-8"))

    def prune_weights(self):
        prunable = []
        for feature, label_weights in self._w.iteritems():
            if all((abs(w) < self.prune_limit for w in label_weights)):
                prunable.append(feature)
        for f in prunable:
            del self._w[f]

    ############################################################################
    #### Standard (independent) prediction #####################################
    ############################################################################

    def _perform_train_iteration_independent(self, x, y, permutation):
        correct, total = 0, len(x)
        for n in range(len(x)):
            if (n % 100) == 0: self._progress(n)
            idx = permutation[n]
            guess = self.predict_features(x[idx])
            truth = y[idx]
            if guess != truth:
                # update step
                vec = np.zeros(len(self._label_mapper))
                vec[self._label_mapper[truth]] = self.learning_rate
                vec[self._label_mapper[guess]] = -1.0 * self.learning_rate
                for feat, value in x[idx].iteritems():
                    self._w[feat] += (value * vec)
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
                features = self._feature_extractor.get(
                    pad_x, pos, history=history
                    )
                guess = self.predict_features(features)
                truth = truth_seq[pos - self._left_context_size]
                if guess != truth:
                    # update step
                    vec = np.zeros(len(self._label_mapper))
                    vec[self._label_mapper[truth]] = self.learning_rate
                    vec[self._label_mapper[guess]] = -1.0 * self.learning_rate
                    for feat, value in features.iteritems():
                        self._w[feat] += (value * vec)
                else:
                    correct += 1
                history.append(guess)
        return (correct, total)
