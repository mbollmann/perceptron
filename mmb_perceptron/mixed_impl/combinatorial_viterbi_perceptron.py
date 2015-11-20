# -*- coding: utf-8 -*-

import itertools as it
import numpy as np
import operator as op
from ..perceptron import Perceptron

class defaultdict_numpy(dict):
    def __init__(self, dim):
        self._dim = dim

    def __missing__(self, key):
        return np.zeros(self._dim)

    def copy(self):
        obj = defaultdict_numpy(self._dim)
        for (key, value) in self.iteritems():
            obj[key] = value.copy()
        return obj

class CombinatorialViterbiPerceptron_Mixed(Perceptron):
    """Perceptron classifier with combinatorial feature explosion and viterbi
    decoding.

    Terribly slow, probably shouldn't be used.
    """

    branch_limit = 8

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

    def predict(self, x):
        """Predict the class label of a given data point.
        """
        if not self.sequenced:
            raise NotImplementedError("viterbi perceptron must be sequenced")

        feature_count = len(self._label_mapper)
        b_lim = self.branch_limit
        (padded_x, history, startpos) = self._initialize_sequence(x)
        current_states = {tuple(history): (0.0, history)}
        future_states = {}

        for i in range(startpos, startpos + len(x)):
            vec_fixed = sum(v * self._w[f] for f, v in \
                            self._feature_extractor.get_fixed(padded_x, i).iteritems())
            for state, (s_prob, s_path) in current_states.iteritems():
                dyn_features = self._feature_extractor.get_dynamic(
                    padded_x, i, history=s_path
                    )
                scores = np.zeros(feature_count) + vec_fixed \
                    + sum(self._w.get(f, 0) * v for f, v in dyn_features.iteritems())
                if feature_count > b_lim:
                    n_best = np.argpartition(scores, -b_lim)[-b_lim:]  # n-best only
                else:
                    n_best = range(feature_count)
                for idx in n_best:
                    label = self._label_mapper.get_name(idx)
                    f_state = state[1:] + (label,)
                    f_prob = s_prob + scores[idx]
                    if f_state in future_states and future_states[f_state][0] > f_prob:
                        continue
                    f_path = s_path + [label]
                    future_states[f_state] = (f_prob, f_path)
            current_states, future_states = future_states, {}

        (best_prob, best_path) = max(current_states.values(), key=op.itemgetter(0))
        guesses = best_path[self._left_context_size:]
        return guesses

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
        self._feature_extractor.init(x)
        new_x = x
        for seq_y in y:
            self._label_mapper.extend(seq_y)
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
        # header
        print("\t" + "\t".join(all_labels).encode("utf-8"))
        sorted_indices = self._label_mapper.map_list(all_labels)
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
    #### Sequenced prediction ##################################################
    ############################################################################

    def _perform_train_iteration_sequenced(self, x, y, permutation):
        correct, total = 0, 0
        for n in range(len(x)):
            idx = permutation[n]
            (pad_x, history, start_pos) = self._initialize_sequence(x[idx])
            truth_seq = y[idx]

            # predict whole sequence first
            history += self.predict(x[idx])
            total += len(x[idx])
            self._progress(total)

            # loop over sequence elements
            for pos in range(start_pos, start_pos + len(x[idx])):
                features = self._feature_extractor.get(
                    pad_x, pos, history=history
                    )
                guess = history[pos]
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
