# -*- coding: utf-8 -*-

import numpy as np
import pytest
from mmb_perceptron.dict_impl import \
     CombinatorialPerceptron as CombinatorialPerceptron_Dict
from mmb_perceptron.numpy_impl import \
     CombinatorialPerceptron as CombinatorialPerceptron_Numpy
from helper_classes import BinaryFeatureExtractor, ContextualFeatureExtractor

perceptron_impls = [CombinatorialPerceptron_Dict, CombinatorialPerceptron_Numpy]

@pytest.mark.parametrize('perceptron', perceptron_impls)
def test_logical_or(perceptron):
    x = [np.array([0,0,1]),
         np.array([0,1,1]),
         np.array([1,0,1]),
         np.array([1,1,1])]
    y = np.array([0, 1, 1, 1])
    p = perceptron(iterations=100)
    p.train(x, y)
    assert p.predict_vector(np.array([0,1,1])) == 1
    assert p.predict_vector(np.array([1,0,1])) == 1
    assert p.predict_vector(np.array([1,1,1])) == 1
    assert p.predict_vector(np.array([0,0,1])) == 0

@pytest.mark.parametrize('perceptron', perceptron_impls)
def test_logical_and(perceptron):
    x = [np.array([0,0,1]),
         np.array([0,1,1]),
         np.array([1,0,1]),
         np.array([1,1,1])]
    y = np.array([0, 0, 0, 1])
    p = perceptron(iterations=100)
    p.train(x, y)
    assert p.predict_vector(np.array([0,1,1])) == 0
    assert p.predict_vector(np.array([1,0,1])) == 0
    assert p.predict_vector(np.array([1,1,1])) == 1
    assert p.predict_vector(np.array([0,0,1])) == 0

@pytest.mark.parametrize('perceptron', perceptron_impls)
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
    p = perceptron(iterations=100)
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

@pytest.mark.parametrize('perceptron', perceptron_impls)
def test_logical_or_with_features(perceptron):
    x = ["False/False", "False/True", "True/False", "True/True"]
    y = ["False", "True", "True", "True"]
    p = perceptron(
            iterations=100,
            feature_extractor=BinaryFeatureExtractor()
        )
    p.train(x, y)
    values = ["True/False", "True/True", "False/False", "False/True"]
    expected = ["True", "True", "False", "True"]
    for v, e in zip(values, expected):
        assert p.predict(v) == e
    assert p.predict_all(values) == expected

@pytest.mark.parametrize('perceptron', perceptron_impls)
def test_logical_or_with_sequence_prediction(perceptron):
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

@pytest.mark.parametrize('perceptron', perceptron_impls)
def test_logical_or_with_sequence_training(perceptron):
    x = [["False/False", "False/True"],
         ["True/False", "True/True"],
         ["True/False", "False/True", "False/False"],
         ["True/True", "False/False"]]
    y = [["False", "True"],
         ["True", "True"],
         ["True", "True", "False"],
         ["True", "False"]]
    p = perceptron(
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

@pytest.mark.parametrize('perceptron', perceptron_impls)
def test_logical_or_with_dynamic_feature_growth(perceptron):
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
            iterations=50,
            feature_extractor=bfe,
            sequenced=True
        )
    p.train(x, y)
    seq = ["False/True", "True/False", "True/True", "False/False"]
    expected = ["True", "True", "True", "False"]
    assert p.predict(seq) == expected

@pytest.mark.parametrize('perceptron', perceptron_impls)
def test_sequenced_number_tagging(perceptron):
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

@pytest.mark.parametrize('perceptron', perceptron_impls)
def test_can_be_pickled(perceptron):
    import pickle
    x = ["False/False", "False/True", "True/False", "True/True"]
    y = ["False", "True", "True", "True"]
    p = perceptron(
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
