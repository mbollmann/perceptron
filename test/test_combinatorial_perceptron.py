# -*- coding: utf-8 -*-

import itertools as it
import numpy as np
import pytest
from mmb_perceptron.dict_impl import \
     CombinatorialPerceptron as CombinatorialPerceptron_Dict
from mmb_perceptron.numpy_impl import \
     CombinatorialPerceptron as CombinatorialPerceptron_Numpy
from mmb_perceptron.mixed_impl import \
     CombinatorialPerceptron as CombinatorialPerceptron_Mixed
from helper_classes import \
     BinaryFeatureExtractor, ContextualFeatureExtractor, \
     ThreeClassesFeatureExtractor
from helper_functions import \
     _train_sequenced_number_tagging

perceptron_numpy_only = [CombinatorialPerceptron_Numpy]
perceptron_impls = [CombinatorialPerceptron_Dict,
                    CombinatorialPerceptron_Numpy,
                    CombinatorialPerceptron_Mixed]

@pytest.mark.parametrize('perceptron', perceptron_numpy_only)
def test_logical_or(perceptron):
    x = [np.array([0,0,1]),
         np.array([0,1,1]),
         np.array([1,0,1]),
         np.array([1,1,1])]
    y = np.array([0, 1, 1, 1])
    # Note: if we're working with vectors directly, we need to disable pruning,
    #       since pruning might change the dimensionality
    p = perceptron(iterations=100, pruning=False)
    p.train(x, y)
    assert p.predict_vector(np.array([0,1,1])) == 1
    assert p.predict_vector(np.array([1,0,1])) == 1
    assert p.predict_vector(np.array([1,1,1])) == 1
    assert p.predict_vector(np.array([0,0,1])) == 0

@pytest.mark.parametrize('perceptron', perceptron_numpy_only)
def test_logical_and(perceptron):
    x = [np.array([0,0,1]),
         np.array([0,1,1]),
         np.array([1,0,1]),
         np.array([1,1,1])]
    y = np.array([0, 0, 0, 1])
    p = perceptron(iterations=100, pruning=False)
    p.train(x, y)
    assert p.predict_vector(np.array([0,1,1])) == 0
    assert p.predict_vector(np.array([1,0,1])) == 0
    assert p.predict_vector(np.array([1,1,1])) == 1
    assert p.predict_vector(np.array([0,0,1])) == 0

@pytest.mark.parametrize('perceptron', perceptron_numpy_only)
def test_three_classes(perceptron):
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
    p = perceptron(iterations=100, pruning=False)
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

@pytest.mark.parametrize('averaged,perceptron',
                         it.product([True, False], perceptron_impls))
def test_logical_or_with_features(averaged, perceptron):
    x = ["False/False", "False/True", "True/False", "True/True"]
    y = ["False", "True", "True", "True"]
    p = perceptron(
            averaged=averaged,
            iterations=100,
            feature_extractor=BinaryFeatureExtractor()
        )
    p.train(x, y)
    values = ["True/False", "True/True", "False/False", "False/True"]
    expected = ["True", "True", "False", "True"]
    print(p._w)
    for v, e in zip(values, expected):
        assert p.predict(v) == e
    assert p.predict_all(values) == expected

@pytest.mark.parametrize('averaged,perceptron',
                         it.product([True, False], perceptron_impls))
def test_logical_and_with_features(averaged, perceptron):
    x = ["False/False", "False/True", "True/False", "True/True"]
    y = ["False", "False", "False", "True"]
    p = perceptron(
            averaged=averaged,
            iterations=100,
            feature_extractor=BinaryFeatureExtractor()
        )
    p.train(x, y)
    values = ["True/False", "True/True", "False/False", "False/True"]
    expected = ["False", "True", "False", "False"]
    for v, e in zip(values, expected):
        assert p.predict(v) == e
    assert p.predict_all(values) == expected

@pytest.mark.parametrize('averaged,perceptron',
                         it.product([True, False], perceptron_impls))
def test_three_classes_with_features(averaged, perceptron):
    x = ["", "b", "d", "bd", "a", "ad", "ac", "c", "bc"]
    y = ["one", "one", "one", "one", "two", "two", "three", "three", "three"]
    p = perceptron(
        averaged=averaged,
        feature_extractor=ThreeClassesFeatureExtractor(),
        iterations=100
        )
    p.train(x, y)
    values = ["foo", "d", "b", "a", "c",
              "db", "ad", "bc", "ac", "abcd"]
    expected = ["one", "one", "one", "two", "three",
                "one", "two", "three", "three", "three"]
    for v, e in zip(values, expected):
        assert p.predict(v) == e
    assert p.predict_all(values) == expected

@pytest.mark.parametrize('averaged,perceptron',
                         it.product([True, False], perceptron_impls))
def test_logical_or_with_sequence_prediction(averaged, perceptron):
    # rationale: sequence prediction should be identical to individual
    # prediction when the feature extractor is oblivious to it
    x = ["False/False", "False/True", "True/False", "True/True"]
    y = ["False", "True", "True", "True"]
    p = perceptron(
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

@pytest.mark.parametrize('averaged,perceptron',
                         it.product([True, False], perceptron_impls))
def test_logical_or_with_sequence_training(averaged, perceptron):
    x = [["False/False", "False/True"],
         ["True/False", "True/True"],
         ["True/False", "False/True", "False/False"],
         ["True/True", "False/False"]]
    y = [["False", "True"],
         ["True", "True"],
         ["True", "True", "False"],
         ["True", "False"]]
    p = perceptron(
            averaged=averaged,
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

@pytest.mark.parametrize('averaged,perceptron',
                         it.product([True, False], perceptron_impls))
def test_logical_or_with_dynamic_feature_growth(averaged, perceptron):
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
    p = perceptron(
            averaged=averaged,
            iterations=50,
            feature_extractor=bfe,
            sequenced=True
        )
    p.train(x, y)
    seq = ["False/True", "True/False", "True/True", "False/False"]
    expected = ["True", "True", "True", "False"]
    assert p.predict(seq) == expected

@pytest.mark.parametrize('averaged,perceptron',
                         it.product([True, False], perceptron_impls))
def test_sequenced_number_tagging(averaged, perceptron):
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
    p = perceptron(
            averaged=averaged,
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

@pytest.mark.parametrize('averaged,perceptron',
                         it.product([True, False], perceptron_impls))
def test_logical_or_after_pickling_and_unpickling(averaged, perceptron):
    import pickle
    x = ["False/False", "False/True", "True/False", "True/True"]
    y = ["False", "True", "True", "True"]
    p = perceptron(
            averaged=averaged,
            iterations=100,
            feature_extractor=BinaryFeatureExtractor()
        )
    p.train(x, y)
    # serialization/unserialization
    serialized = pickle.dumps(p)
    del p
    p = pickle.loads(serialized)
    # test if everything still works
    values = ["True/False", "True/True", "False/False", "False/True"]
    expected = ["True", "True", "False", "True"]
    for v, e in zip(values, expected):
        assert p.predict(v) == e
    assert p.predict_all(values) == expected

@pytest.mark.parametrize('averaged,perceptron',
                         it.product([True, False], perceptron_impls))
def test_sequenced_number_tagging_after_pickling_and_unpickling(averaged, perceptron):
    """Dumb sequence tagging example: Perceptron learns to tag numbers with
    their respective string, except for 2 following 1 which is tagged
    'twelve'.
    """
    import pickle
    p = _train_sequenced_number_tagging(averaged, perceptron,
                                        ContextualFeatureExtractor())
    # serialization/unserialization
    serialized = pickle.dumps(p)
    del p
    p = pickle.loads(serialized)
    # test if everything still works
    sequences = [["0", "1", "2"],
                 ["1", "0", "2", "1", "2", "2", "2"],
                 ["2", "1", "1", "2", "0"]]
    expected = [["ZERO", "ONE", "TWELVE"],
                ["ONE", "ZERO", "TWO", "ONE", "TWELVE", "TWO", "TWO"],
                ["TWO", "ONE", "ONE", "TWELVE", "ZERO"]]
    for s, e in zip(sequences, expected):
        assert p.predict(s) == e
    assert p.predict_all(sequences) == expected
