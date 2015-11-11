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


class ContextualFeatureExtractor(FeatureExtractor):
    _context_size = (1, 1)

    def _init_sequenced(self, dataset):
        pass

    def _get_sequenced(self, seq, pos, history=None):
        features = {}
        features['this==' + seq[pos]] = 1.0
        features['prev==' + seq[pos - 1]] = 1.0
        features['next==' + seq[pos + 1]] = 1.0
        features['prev_guess==' + history[pos - 1]] = 1.0
        return features


class NumberFeatureGenerator(GenerativeExtractor):
    def _init_independent(self, dataset):
        self._label_mapper.extend(range(6))

    def _generate_vector_independent(self, x, truth=None):
        (a, b) = x
        f_false = np.array([a,b,0,0,0,1])
        f_true = np.array([0,0,a,b,1,0])
        if truth == 1:
            return (np.array([f_true, f_false]), [1, 0])
        else:
            return (np.array([f_false, f_true]), [0, 1])