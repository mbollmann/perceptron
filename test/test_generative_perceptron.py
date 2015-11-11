# -*- coding: utf-8 -*-

import numpy as np
from mmb_perceptron import GenerativePerceptron
from helper_classes import BinaryFeatureGenerator, NumberFeatureGenerator

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
