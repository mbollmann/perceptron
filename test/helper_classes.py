# -*- coding: utf-8 -*-
"""Contains helper classes, such as feature extractors, that are used in the tests.
"""

import numpy as np
from mmb_perceptron.feature_extractor import FeatureExtractor
from mmb_perceptron.feature_extractor.generator import GenerativeExtractor

class BinaryFeatureExtractor(FeatureExtractor):
    _binary_featureset = ('bias', 'lhs_true', 'rhs_true')

    def _init_independent(self, dataset):
        self._label_mapper.extend(self._binary_featureset)
    _init_sequenced = _init_independent

    def _get_independent(self, x):
        features = {'bias': 1.0}
        if x.startswith("True"):
            features['lhs_true'] = 1.0
        if x.endswith("True"):
            features['rhs_true'] = 1.0
        return features

    def _get_sequenced(self, seq, pos, history=None):
        return self._get_independent(seq[pos])


class BinaryFeatureGenerator(BinaryFeatureExtractor, GenerativeExtractor):
    def _generate_independent(self, x, truth=None):
        a = 1 if x.startswith("True") else 0
        b = 1 if x.endswith("True") else 0
        f_false = {
            'bias && false': 1.0,
            'lhs_true && false': 1.0 if a == 1 else 0.0,
            'rhs_true && false': 1.0 if b == 1 else 0.0
            }
        f_true = {
            'bias && true': 1.0,
            'lhs_true && true': 1.0 if a == 1 else 0.0,
            'rhs_true && true': 1.0 if b == 1 else 0.0
            }
        if truth == 'True':
            return ([f_true, f_false], ['True', 'False'])
        else:
            return ([f_false, f_true], ['False', 'True'])


class CharacterLengthGenerator(GenerativeExtractor):
    def _generate_independent(self, x, truth=None):
        features, labels = [], []
        for d in range(1, 5):
            label = '{0}x{1}'.format(d, x[0])
            feature = {'inputlength==d': 1.0 if len(x) == d else 0.0,
                       'length_{0}'.format(d): 1.0,
                       'bias_{0}'.format(label): 1.0}
            insert_pos = 0 if label == truth else len(features)
            features.insert(insert_pos, feature)
            labels.insert(insert_pos, label)
        return (features, labels)


class ContextualFeatureExtractor(FeatureExtractor):
    _left_context_size = 1
    _right_context_size = 1

    def _get_sequenced(self, seq, pos, history=None):
        features = {'bias': 1.0}
        features['this==' + seq[pos]] = 1.0
        features['prev==' + seq[pos - 1]] = 1.0
        features['next==' + seq[pos + 1]] = 1.0
        features['prev_guess==' + history[pos - 1]] = 1.0
        return features


class ContextualFeatureGenerator(ContextualFeatureExtractor, GenerativeExtractor):
    _all_labels = ["ONE", "TWO", "TWELVE", "ZERO"]

    def _generate_sequenced(self, seq, pos, history=None, truth=None):
        features, labels = [], []
        old_feats = self._get_sequenced(seq, pos, history=history)
        # Combinatorial feature explosion -- i.e., we mimic what the
        # CombinatorialPerceptron does internally:
        for label in self._all_labels:
            feats = {'{0} && label={1}'.format(f, label): 1.0 for f in old_feats}
            insert_pos = 0 if label == truth else len(features)
            features.insert(insert_pos, feats)
            labels.insert(insert_pos, label)
        return (features, labels)


class NumberFeatureGenerator(GenerativeExtractor):
    def _init_independent(self, dataset):
        self._label_mapper.extend(range(6))

    def _generate_independent(self, x, truth=None):
        (a, b) = x
        f_false = {
            'a && false': a,
            'b && false': b,
            'bias && false': 1.0
            }
        f_true = {
            'a && true': a,
            'b && true': b,
            'bias && true': 1.0
            }
        if truth == 1:
            return ([f_true, f_false], [1, 0])
        else:
            return ([f_false, f_true], [0, 1])

    def _generate_vector_independent(self, x, truth=None):
        (a, b) = x
        f_false = np.array([a,b,0,0,0,1])
        f_true = np.array([0,0,a,b,1,0])
        if truth == 1:
            return (np.array([f_true, f_false]), [1, 0])
        else:
            return (np.array([f_false, f_true]), [0, 1])


class ThreeClassesFeatureExtractor(FeatureExtractor):
    _featureset = ('bias', 'a', 'b', 'c', 'd')

    def _init_independent(self, dataset):
        self._label_mapper.extend(self._featureset)
    _init_sequenced = _init_independent

    def _get_independent(self, x):
        features = {'bias': 1.0}
        for feat in self._featureset:
            if feat in x:
                features[feat] = 1.0
        return features

    def _get_sequenced(self, seq, pos, history=None):
        return self._get_independent(seq[pos])

