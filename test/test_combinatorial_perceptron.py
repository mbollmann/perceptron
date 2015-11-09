# -*- coding: utf-8 -*-

import numpy as np
from mmb_perceptron import CombinatorialPerceptron
from mmb_perceptron.feature_extractor import FeatureExtractor

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


class TestCombinatorialPerceptron(object):
    """Tests for the combinatorial perceptron.

    Contains tests that primarily test the perceptron with "sequenced=False".
    Includes some tests with "sequenced=True" as well, though the feature
    extractor and training data are always non-sequential in nature.
    """

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
        p.sequenced = True
        seq = ["False/True", "True/False", "True/True", "False/False"]
        expected = ["True", "True", "True", "False"]
        assert p.predict(seq) == expected

    def test_logical_or_with_sequence_training(self):
        x = [["False/False", "False/True"],
             ["True/False", "True/True"],
             ["True/False", "False/True", "False/False"],
             ["True/True", "False/False"]]
        y = [["False", "True"],
             ["True", "True"],
             ["True", "True", "False"],
             ["True", "False"]]
        p = CombinatorialPerceptron(
                iterations=50,
                feature_extractor=BinaryFeatureExtractor(),
                sequenced=True
            )
        p.train(x, y)
        seq = ["False/True", "True/False", "True/True", "False/False"]
        expected = ["True", "True", "True", "False"]
        assert p.predict(seq) == expected

    def test_logical_or_with_dynamic_feature_growth(self):
        x = [["False/False", "False/True"],
             ["True/False", "True/True"],
             ["True/False", "False/True", "False/False"],
             ["True/True", "False/False"]]
        y = [["False", "True"],
             ["True", "True"],
             ["True", "True", "False"],
             ["True", "False"]]
        bfe = BinaryFeatureExtractor()
        # only start with the bias feature, and let the other ones be grown
        # automatically over time
        bfe._binary_featureset = ('bias',)
        p = CombinatorialPerceptron(
                iterations=50,
                feature_extractor=bfe,
                sequenced=True
            )
        p.train(x, y)
        seq = ["False/True", "True/False", "True/True", "False/False"]
        expected = ["True", "True", "True", "False"]
        assert p.predict(seq) == expected
