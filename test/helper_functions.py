# -*- coding: utf-8 -*-
"""Contains helper functions that are used in the tests.
"""

def _train_sequenced_number_tagging(averaged, perceptron, feature):
    x = [["0", "2", "1"],
         ["0", "1", "2"],
         ["1", "2", "2"],
         ["2", "1", "2"],
         ["1", "1", "1"],
         ["2", "2", "2"],
         ["1", "2", "0"],
         ["1", "0", "2"]]
    y = [["ZERO", "TWO", "ONE"],
         ["ZERO", "ONE", "TWELVE"],
         ["ONE", "TWELVE", "TWO"],
         ["TWO", "ONE", "TWELVE"],
         ["ONE", "ONE", "ONE"],
         ["TWO", "TWO", "TWO"],
         ["ONE", "TWELVE", "ZERO"],
         ["ONE", "ZERO", "TWO"]]
    p = perceptron(
            averaged=averaged,
            iterations=100,
            sequenced=True,
            feature_extractor = feature
        )
    p.train(x, y)
    return p
