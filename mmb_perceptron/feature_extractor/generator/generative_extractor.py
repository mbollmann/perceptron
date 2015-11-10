# -*- coding: utf-8 -*-

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
            self.generate_with_oracle = self._generate_with_oracle_sequenced
            self.generate_vector_with_oracle = \
                self._generate_vector_with_oracle_sequenced
        else:
            self.generate = self._generate_independent
            self.generate_vector = self._generate_vector_independent
            self.generate_with_oracle = self._generate_with_oracle_independent
            self.generate_vector_with_oracle = \
                self._generate_vector_with_oracle_independent

    def _generate_independent(self, x):
        """Return candidates and their feature representations.

        Should return a dict that maps class labels (= candidates) to feature
        representations.
        """
        raise NotImplementedError("function not implemented")

    def _generate_sequenced(self, seq, pos, history=None):
        raise NotImplementedError("function not implemented")

    def _generate_with_oracle_independent(self, x, truth=None):
        """Return candidates, their feature representations, and the oracle-best candidate.

        Should return a tuple (F, C), where F is the dictionary of feature
        representations, and C is an oracle prediction of the best candidate.  C
        is required to be contained in F.

        'truth' can be given to indicate the gold-standard best candidate, but
        it is up to the generator function whether to return this value as its
        prediction C (and include it in the feature representations), or whether
        to generate the candidates independently and use an oracle to select
        the most likely truth candidate from those.
        """
        raise NotImplementedError("function not implemented")

    def _generate_with_oracle_sequenced(self, seq, pos, history=None, truth=None):
        raise NotImplementedError("function not implemented")

    def _generate_vector_independent(self, x):
        predictions = self._generate_independent(x)
        return {l: self._label_mapper.map_to_vector(v) \
                for l, v in predictions.iteritems()}

    def _generate_vector_sequenced(self, seq, pos, history=None):
        predictions = self._generate_sequenced(seq, pos, history=history)
        return {l: self._label_mapper.map_to_vector(v) \
                for l, v in predictions.iteritems()}

    def _generate_vector_with_oracle_independent(self, x, truth=None):
        (predictions, oracle) = self._generate_with_oracle_independent(x, truth=truth)
        return ({l: self._label_mapper.map_to_vector(v) \
                 for l, v in predictions.iteritems()}, oracle)

    def _generate_vector_with_oracle_sequenced(self, seq, pos, history=None, truth=None):
        (predictions, oracle) = self._generate_with_oracle_sequenced(seq, pos,
                                                                     history=history,
                                                                     truth=truth)
        return ({l: self._label_mapper.map_to_vector(v) \
                 for l, v in predictions.iteritems()}, oracle)
