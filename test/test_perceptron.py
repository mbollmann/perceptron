# -*- coding: utf-8 -*-

import numpy as np
from mmb_perceptron import CombinatorialPerceptron
from mmb_perceptron.feature_extractor import FeatureExtractor

class BinaryFeatureExtractor(FeatureExtractor):
    def init(self, dataset):
        self._label_mapper.extend(('bias', 'lhs_true', 'rhs_true'))

    def get(self, x):
        features = {'bias': 1.0}
        if x.startswith("True"):
            features['lhs_true'] = 1.0
        if x.endswith("True"):
            features['rhs_true'] = 1.0
        return features

class TestCombinatorialPerceptron(object):
    def test_logical_or(self):
        x = [np.array([0,0,1]),
             np.array([0,1,1]),
             np.array([1,0,1]),
             np.array([1,1,1])]
        y = np.array([0, 1, 1, 1])
        p = CombinatorialPerceptron(iterations=100)
        p.train(x, y)
        assert p.predict_vector(np.array([0,1,1])) == 1
        assert p.predict_vector(np.array([1,0,1])) == 1
        assert p.predict_vector(np.array([1,1,1])) == 1
        assert p.predict_vector(np.array([0,0,1])) == 0

    def test_logical_and(self):
        x = [np.array([0,0,1]),
             np.array([0,1,1]),
             np.array([1,0,1]),
             np.array([1,1,1])]
        y = np.array([0, 0, 0, 1])
        p = CombinatorialPerceptron(iterations=100)
        p.train(x, y)
        assert p.predict_vector(np.array([0,1,1])) == 0
        assert p.predict_vector(np.array([1,0,1])) == 0
        assert p.predict_vector(np.array([1,1,1])) == 1
        assert p.predict_vector(np.array([0,0,1])) == 0

    def test_three_classes(self):
        x = [np.array([0,0,0,0,1]),
             np.array([0,1,0,0,1]),
             np.array([0,0,0,1,1]),
             np.array([0,1,0,1,1]),
             np.array([1,0,0,0,1]),
             np.array([1,0,0,1,1]),
             np.array([1,0,1,0,1]),
             np.array([0,0,1,0,1]),
             np.array([0,1,1,0,1])
             ]
        y = np.array([0,0,0,0,1,1,2,2,2])
        p = CombinatorialPerceptron(iterations=100)
        p.train(x, y)
        assert p.predict_vector(np.array([0,0,0,0,1])) == 0
        assert p.predict_vector(np.array([0,0,0,1,1])) == 0
        assert p.predict_vector(np.array([0,1,0,0,1])) == 0
        assert p.predict_vector(np.array([1,0,0,0,1])) == 1
        assert p.predict_vector(np.array([0,0,1,0,1])) == 2
        assert p.predict_vector(np.array([0,1,0,1,1])) == 0
        assert p.predict_vector(np.array([1,0,0,1,1])) == 1
        assert p.predict_vector(np.array([0,1,1,0,1])) == 2
        assert p.predict_vector(np.array([1,0,1,0,1])) == 2
        assert p.predict_vector(np.array([1,1,1,1,1])) == 2

    def test_logical_or_with_features(self):
        x = ["False/False", "False/True", "True/False", "True/True"]
        y = ["False", "True", "True", "True"]
        p = CombinatorialPerceptron(
                iterations=100,
                feature_extractor=BinaryFeatureExtractor()
            )
        p.train(x, y)
        assert p.predict("False/True") == "True"
        assert p.predict("True/False") == "True"
        assert p.predict("True/True") == "True"
        assert p.predict("False/False") == "False"

    def test_logical_or_with_sequence_prediction(self):
        # rationale: sequence prediction should be identical to individual
        # prediction when the feature extractor is oblivious to it
        x = ["False/False", "False/True", "True/False", "True/True"]
        y = ["False", "True", "True", "True"]
        p = CombinatorialPerceptron(
                iterations=100,
                feature_extractor=BinaryFeatureExtractor()
            )
        p.train(x, y)
        seq = ["False/True", "True/False", "True/True", "False/False"]
        expected = ["True", "True", "True", "False"]
        assert p.predict_sequence(seq) == expected
