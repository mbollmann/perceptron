# -*- coding: utf-8 -*-

import numpy as np
from mmb_perceptron import CombinatorialPerceptron
from helper_classes import BinaryFeatureExtractor, ContextualFeatureExtractor

class TestCombinatorialPerceptron(object):
    """Tests for the combinatorial perceptron.
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
        values = ["True/False", "True/True", "False/False", "False/True"]
        expected = ["True", "True", "False", "True"]
        assert p.predict_all(values) == expected

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
        sequences = [["False/True", "True/False", "True/True", "False/False"],
                     ["False/False", "False/False"],
                     ["True/False", "True/True"]]
        expected = [["True", "True", "True", "False"],
                    ["False", "False"],
                    ["True", "True"]]
        for s, e in zip(sequences, expected):
            assert p.predict(s) == e
        assert p.predict_all(sequences) == expected

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
        sequences = [["False/True", "True/False", "True/True", "False/False"],
                     ["False/False", "False/False"],
                     ["True/False", "True/True"]]
        expected = [["True", "True", "True", "False"],
                    ["False", "False"],
                    ["True", "True"]]
        for s, e in zip(sequences, expected):
            assert p.predict(s) == e
        assert p.predict_all(sequences) == expected

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

    def test_sequenced_number_tagging(self):
        """Dumb sequence tagging example: Perceptron learns to tag numbers with
        their respective string, except for 2 following 1 which is tagged
        'twelve'.
        """
        x = [["0", "2", "1"],
             ["0", "1", "2"],
             ["1", "2", "2"],
             ["2", "1", "2"],
             ["1", "1", "1"],
             ["2", "2", "2"],
             ["1", "0", "2"]]
        y = [["ZERO", "TWO", "ONE"],
             ["ZERO", "ONE", "TWELVE"],
             ["ONE", "TWELVE", "TWO"],
             ["TWO", "ONE", "TWELVE"],
             ["ONE", "ONE", "ONE"],
             ["TWO", "TWO", "TWO"],
             ["ONE", "ZERO", "TWO"]]
        p = CombinatorialPerceptron(
            iterations=100,
            sequenced=True,
            feature_extractor = ContextualFeatureExtractor()
            )
        p.train(x, y)
        sequences = [["0", "1", "2"],
                     ["1", "0", "2", "1", "2", "2", "2"],
                     ["2", "1", "1", "2", "0"]]
        expected = [["ZERO", "ONE", "TWELVE"],
                    ["ONE", "ZERO", "TWO", "ONE", "TWELVE", "TWO", "TWO"],
                    ["TWO", "ONE", "ONE", "TWELVE", "ZERO"]]
        for s, e in zip(sequences, expected):
            assert p.predict(s) == e
        assert p.predict_all(sequences) == expected
