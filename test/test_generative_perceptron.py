# -*- coding: utf-8 -*-

import numpy as np
from mmb_perceptron import GenerativePerceptron
from mmb_perceptron.feature_extractor.generator import GenerativeExtractor
from test_combinatorial_perceptron import BinaryFeatureExtractor

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


class TestGenerativePerceptron(object):
    """Tests for the generative perceptron.
    """

    def test_logical_or(self):
        x = [(0,0), (0,1), (1,0), (1,1)]
        y = [0, 1, 1, 1]
        p = GenerativePerceptron(
            feature_extractor = NumberFeatureGenerator(),
            iterations = 100
            )
        p.train(x, y)
        assert p.predict((0,1)) == 1
        assert p.predict((1,0)) == 1
        assert p.predict((1,1)) == 1
        assert p.predict((0,0)) == 0

    def test_logical_and(self):
        x = [(0,0), (0,1), (1,0), (1,1)]
        y = [0, 0, 0, 1]
        p = GenerativePerceptron(
            feature_extractor = NumberFeatureGenerator(),
            iterations = 100
            )
        p.train(x, y)
        assert p.predict((0,1)) == 0
        assert p.predict((1,0)) == 0
        assert p.predict((1,1)) == 1
        assert p.predict((0,0)) == 0

    def test_logical_or_with_features(self):
        x = ["False/False", "False/True", "True/False", "True/True"]
        y = ["False", "True", "True", "True"]
        p = GenerativePerceptron(
                feature_extractor = BinaryFeatureGenerator(),
                iterations = 100
            )
        p.train(x, y)
        assert p.predict("False/True") == "True"
        assert p.predict("True/False") == "True"
        assert p.predict("True/True") == "True"
        assert p.predict("False/False") == "False"
