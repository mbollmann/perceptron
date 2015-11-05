# -*- coding: utf-8 -*-

import numpy as np
from mmb_perceptron import CombinatorialPerceptron

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
