# -*- coding: utf-8 -*-

import numpy as np
from mmb_perceptron import GenerativePerceptron
from mmb_perceptron.feature_extractor.generator import GenerativeExtractor
from test_combinatorial_perceptron import BinaryFeatureExtractor

class NumberFeatureGenerator(GenerativeExtractor):
    def _generate_independent(self, x):
        (a, b) = x
        return {
            0: np.array([a,b,0,0,1]),
            1: np.array([0,0,a,b,1])
            }

    def _generate_with_oracle_independent(self, x, truth=None):
        return (self._generate_independent(x),
                truth if truth is not None else 1)

class BinaryFeatureGenerator(BinaryFeatureExtractor):
    pass


class TestGenerativePerceptron(object):
    """Tests for the generative perceptron.
    """

    def test_logical_or(self):
        x = [(0,0), (0,1), (1,0), (1,1)]
        y = [0, 1, 1, 1]
        p = GenerativePerceptron(
            feature_extractor = NumberFeatureGenerator(),
            iterations=100
            )
        p.train(x, y)
        assert p.predict((0,1)) == 1
        assert p.predict((1,0)) == 1
        assert p.predict((1,1)) == 1
        assert p.predict((0,0)) == 0
