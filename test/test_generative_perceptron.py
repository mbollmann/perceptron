# -*- coding: utf-8 -*-

import numpy as np
from mmb_perceptron import GenerativePerceptron
from helper_classes import \
     BinaryFeatureGenerator, CharacterLengthGenerator, NumberFeatureGenerator

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
        values = ["False/True", "True/False", "True/True", "False/False"]
        expected = ["True", "True", "True", "False"]
        for v, e in zip(values, expected):
            assert p.predict(v) == e
        assert p.predict_all(values) == expected

    def test_character_length(self):
        x = ["A", "AA", "AAA", "AAAA",
             "B", "BB", "BBB", "BBBB",
             "C", "CCC", "DDDD"]
        y = ["1xA", "2xA", "3xA", "4xA",
             "1xB", "2xB", "3xB", "4xB",
             "1xC", "3xC", "4xD"]
        p = GenerativePerceptron(
                feature_extractor = CharacterLengthGenerator(),
                iterations = 50
            )
        p.train(x, y)
        values = ["A", "AA", "AAA", "AAAA", "BBB", "CC", "D", "XXX"]
        expected = ["1xA", "2xA", "3xA", "4xA", "3xB", "2xC", "1xD", "3xX"]
        for v, e in zip(values, expected):
            assert p.predict(v) == e
        assert p.predict_all(values) == expected
