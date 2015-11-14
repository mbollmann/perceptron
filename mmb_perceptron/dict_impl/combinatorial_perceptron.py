# -*- coding: utf-8 -*-

from collections import defaultdict
import itertools as it
import operator as op
from ..perceptron import Perceptron

def defaultdict_float(): return defaultdict(float)

class CombinatorialPerceptron_Dict(Perceptron):
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
        self._w = defaultdict(defaultdict_float)

    def _resize_weights(self, w):
        pass

    def predict_features(self, features):
        scores = defaultdict(float)
        for feat, value in features.iteritems():
            if value == 0 or feat not in self._w:
                continue
            for label, weight in self._w[feat].iteritems():
                scores[label] += value * weight
        # Note: In case of multiple maximum values, it's important that the
        # return prediction remains consistent; that's why we use all_labels
        # here as basis for the prediction, which is a **sorted** list of class
        # labels.
        return max(self.all_labels, key=lambda label: scores[label])

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
        averaged = defaultdict(defaultdict_float)
        divisor = float(len(all_w))
        final_w = all_w[-1]
        for feat, label_weights in final_w.iteritems():
            for label, weight in label_weights.iteritems():
                averaged[feat][label] = \
                    sum((_w[feat][label] for _w in all_w)) / divisor
        return averaged

    ############################################################################
    #### Standard (independent) prediction #####################################
    ############################################################################

    def _perform_train_iteration_independent(self, x, y, permutation):
        correct, total = 0, len(x)
        for n in range(len(x)):
            idx = permutation[n]
            guess = self.predict_features(x[idx])
            truth = y[idx]
            if guess != truth:
                # update step
                for feat, value in x[idx].iteritems():
                    self._w[feat][truth] += self.learning_rate * value
                    self._w[feat][guess] -= self.learning_rate * value
            else:
                correct += 1
        return 1.0 * correct / total

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
                features = self._feature_extractor.get(
                    pad_x, pos, history=history
                    )
                guess = self.predict_features(features)
                truth = truth_seq[pos - self._left_context_size]
                if guess != truth:
                    # update step
                    for feat, value in features.iteritems():
                        self._w[feat][truth] += self.learning_rate * value
                        self._w[feat][guess] -= self.learning_rate * value
                else:
                    correct += 1
                history.append(guess)

        return 1.0 * correct / total
