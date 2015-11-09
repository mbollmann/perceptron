# -*- coding: utf-8 -*-

from mmb_perceptron import CombinatorialPerceptron
from mmb_perceptron.feature_extractor import FeatureExtractor

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

class TestSequencedPerceptron(object):
    """Tests for perceptrons with sequenced=True.
    """

    def test_number_tagging(self):
        """Dumb sequence tagging example: Perceptron learns to tag numbers with
        their respective string, except for 2 following 1 which is tagged
        'twelve'.
        """
        x = [["0", "2", "1"],
             ["0", "1", "2"],
             ["1", "2", "2"],
             ["2", "1", "2"],
             ["1", "0", "2"]]
        y = [["ZERO", "TWO", "ONE"],
             ["ZERO", "ONE", "TWELVE"],
             ["ONE", "TWELVE", "TWO"],
             ["TWO", "ONE", "TWELVE"],
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
