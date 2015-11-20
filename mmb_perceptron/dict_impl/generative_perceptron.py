# -*- coding: utf-8 -*-

from collections import defaultdict
import itertools as it
import numpy as np
from ..perceptron import Perceptron

class GenerativePerceptron_Dict(Perceptron):
    """Perceptron classifier with generation function.

    This classifier uses a generation function to get a list of prediction
    candidates for each data point.  These are then fed into the feature
    extractor, which is expected to generate a feature vector based on both the
    data point and the candidate prediction.  Perceptron weights are a vector,
    which is multiplied with the feature vector to obtain a score, and the best
    score from the generated feature vectors determines the prediction.
    """

    def reset_weights(self):
        self._w = defaultdict(float)

    def _resize_weights(self, w):
        pass

    def _preprocess_train(self, x, y):
        assert len(x) == len(y)
        self._feature_extractor.init(x)
        self._label_mapper.reset()
        return (x, y)

    def predict_features(self, features):
        """Predicts the best feature vector from a list of feature vectors.
        """
        def make_score(feats):
            return sum((self._w.get(f, 0) * v for (f, v) in feats.iteritems()))

        scores = [make_score(feats) for feats in features]
        s_best = max(scores)
        p_best = [n for (n, s) in enumerate(scores) if s == s_best]
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
                (features, labels) = self._feature_extractor.generate(
                    padded_x, i, history=history
                    )
                guess = labels[self.predict_features(features)]
                history.append(guess)
            guesses = history[self._left_context_size:]
            return guesses
        else:
            (features, labels) = self._feature_extractor.generate(x)
            guess = labels[self.predict_features(features)]
            return guess

    def average_weights(self, all_w):
        averaged = defaultdict(float)
        divisor = float(len(all_w))
        final_w = all_w[-1]
        for feat, weight in final_w.iteritems():
            averaged[feat] = sum((_w[feat] for _w in all_w)) / divisor
        return averaged

    def print_weights(self):
        # feature weights
        for feature, weight in self._w.iteritems():
            print("\t".join((feature, unicode(weight))).encode("utf-8"))

    def prune_weights(self):
        prunable = [f for f, w in self._w.iteritems() if abs(w) < self.prune_limit]
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
            (features, _) = \
                self._feature_extractor.generate(x[idx], truth=y[idx])
            guess = self.predict_features(features)
            if guess != 0:
                # update step
                for feat, value in features[0].iteritems():
                    self._w[feat] += self.learning_rate * value
                for feat, value in features[guess].iteritems():
                    self._w[feat] -= self.learning_rate * value
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
                (features, labels) = self._feature_extractor.generate(
                    pad_x, pos, history=history,
                    truth=truth_seq[pos - self._left_context_size]
                    )
                guess = self.predict_features(features)
                if guess != 0:
                    # update step
                    for feat, value in features[0].iteritems():
                        self._w[feat] += self.learning_rate * value
                    for feat, value in features[guess].iteritems():
                        self._w[feat] -= self.learning_rate * value
                else:
                    correct += 1
                history.append(labels[guess])
        return (correct, total)
