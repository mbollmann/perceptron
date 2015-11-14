# -*- coding: utf-8 -*-

import numpy as np
import pytest
from mmb_perceptron.dict_impl import \
     GenerativePerceptron as GenerativePerceptron_Dict
from mmb_perceptron.numpy_impl import \
     GenerativePerceptron as GenerativePerceptron_Numpy
from helper_classes import \
     BinaryFeatureGenerator, CharacterLengthGenerator, \
     ContextualFeatureGenerator, NumberFeatureGenerator

perceptron_impls = [GenerativePerceptron_Dict, GenerativePerceptron_Numpy]

@pytest.mark.parametrize('perceptron', perceptron_impls)
def test_logical_or(perceptron):
    x = [(0,0), (0,1), (1,0), (1,1)]
    y = [0, 1, 1, 1]
    p = perceptron(
        feature_extractor = NumberFeatureGenerator(),
        iterations = 100
        )
    p.train(x, y)
    values = [(0,1), (1,0), (1,1), (0,0)]
    expected = [1, 1, 1, 0]
    for v, e in zip(values, expected):
        assert p.predict(v) == e
    assert p.predict_all(values) == expected

@pytest.mark.parametrize('perceptron', perceptron_impls)
def test_logical_and(perceptron):
    x = [(0,0), (0,1), (1,0), (1,1)]
    y = [0, 0, 0, 1]
    p = perceptron(
        feature_extractor = NumberFeatureGenerator(),
        iterations = 100
        )
    p.train(x, y)
    values = [(0,1), (1,0), (1,1), (0,0)]
    expected = [0, 0, 1, 0]
    for v, e in zip(values, expected):
        assert p.predict(v) == e
    assert p.predict_all(values) == expected

@pytest.mark.parametrize('perceptron', perceptron_impls)
def test_logical_or_with_features(perceptron):
    x = ["False/False", "False/True", "True/False", "True/True"]
    y = ["False", "True", "True", "True"]
    p = perceptron(
            feature_extractor = BinaryFeatureGenerator(),
            iterations = 100
        )
    p.train(x, y)
    values = ["False/True", "True/False", "True/True", "False/False"]
    expected = ["True", "True", "True", "False"]
    for v, e in zip(values, expected):
        assert p.predict(v) == e
    assert p.predict_all(values) == expected

@pytest.mark.parametrize('perceptron', perceptron_impls)
def test_character_length_tagging(perceptron):
    x = ["A", "AA", "AAA", "AAAA",
         "B", "BB", "BBB", "BBBB",
         "C", "CCC", "DDDD"]
    y = ["1xA", "2xA", "3xA", "4xA",
         "1xB", "2xB", "3xB", "4xB",
         "1xC", "3xC", "4xD"]
    p = perceptron(
            feature_extractor = CharacterLengthGenerator(),
            iterations = 50
        )
    p.train(x, y)
    values = ["A", "AA", "AAA", "AAAA", "BBB", "CC", "D", "XXX"]
    expected = ["1xA", "2xA", "3xA", "4xA", "3xB", "2xC", "1xD", "3xX"]
    for v, e in zip(values, expected):
        assert p.predict(v) == e
    assert p.predict_all(values) == expected

@pytest.mark.parametrize('perceptron', perceptron_impls)
def test_sequenced_number_tagging(perceptron):
    """Tests that GenerativePerceptron with a feature extractor doing
    combinatorial feature explosion is really equivalent to
    CombinatorialPerceptron.
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
    p = perceptron(
        iterations = 100,
        sequenced = True,
        feature_extractor = ContextualFeatureGenerator()
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

@pytest.mark.parametrize('perceptron', perceptron_impls)
def test_can_be_pickled(perceptron):
    import pickle
    x = ["False/False", "False/True", "True/False", "True/True"]
    y = ["False", "True", "True", "True"]
    p = perceptron(
            feature_extractor = BinaryFeatureGenerator(),
            iterations = 100
        )
    p.train(x, y)
    # serialization/unserialization
    serialized = pickle.dumps(p)
    del p
    p = pickle.loads(serialized)
    # test if everything still works
    values = ["False/True", "True/False", "True/True", "False/False"]
    expected = ["True", "True", "True", "False"]
    for v, e in zip(values, expected):
        assert p.predict(v) == e
    assert p.predict_all(values) == expected
