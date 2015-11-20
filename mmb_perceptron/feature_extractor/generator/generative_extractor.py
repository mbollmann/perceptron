# -*- coding: utf-8 -*-

import numpy as np
from .. import FeatureExtractor

class GenerativeExtractor(FeatureExtractor):
    """Abstract base class for a generative feature extractor.

    Compared to simple feature extractors, generators perform the additional
    task of generating class label candidates.  This means that they don't
    return a single feature vector, but a dictionary mapping candidate classes
    (for the classifier) to their respective feature vectors.

    In terms of the perceptron algorithm, they combine the GEN() and Phi()
    functions in a single object for ease of implementation.
    """

    def _rebind_methods(self, status):
        super(GenerativeExtractor, self)._rebind_methods(status)
        if status:
            self.generate = self._generate_sequenced
            self.generate_vector = self._generate_vector_sequenced
        else:
            self.generate = self._generate_independent
            self.generate_vector = self._generate_vector_independent

    def _generate_independent(self, x, truth=None):
        """Return candidates and their feature representations.

        Should return a tuple (F, C), where F is a list of feature
        representations, and C is a list of class labels so that C[i] is the
        class label belonging to the feature representation F[i].

        During training, the **first element in these lists** is considered by
        the perceptron to be the **correct class label** for this data point.

        If the parameter 'truth' is supplied, it indicates the gold-standard
        best candidate according to the training data; however, it is up to the
        generator function whether to include this value as the first element of
        the feature representations (thereby making the **gold standard** the
        correct class label for the perceptron learner) or generate the
        candidates independently and select an **oracle-best** class label from
        those.
        """
        raise NotImplementedError("function not implemented")

    def _generate_sequenced(self, seq, pos, history=None, truth=None):
        raise NotImplementedError("function not implemented")

    def _generate_vector_independent(self, x, truth=None, grow=True):
        """Return candidates and their feature representations.

        Identical to _generate_independent(), except that F is now a matrix of
        numerical feature vectors.
        """
        (features, labels) = self._generate_independent(x, truth=truth)
        if grow:
            for f in features:
                self._label_mapper.extend(f)
            vectors = np.array([self._label_mapper.map_to_vector(f) for f in features])
        else:
            vectors = np.array([self._label_mapper.get_vector(f) for f in features])
        return (vectors, labels)

    def _generate_vector_sequenced(self, seq, pos, history=None, truth=None, grow=True):
        (features, labels) = \
            self._generate_sequenced(seq, pos, history=history, truth=truth)
        if grow:
            for f in features:
                self._label_mapper.extend(f)
            vectors = np.array([self._label_mapper.map_to_vector(f) for f in features])
        else:
            vectors = np.array([self._label_mapper.get_vector(f) for f in features])
        return (vectors, labels)
