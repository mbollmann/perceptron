# -*- coding: utf-8 -*-

from ..label_mapper import LabelMapper

class FeatureExtractor(object):
    """Abstract base class for a feature extractor.

    Feature extractors take (potentially arbitrary) data points as input and
    convert them into feature representations, in the form of dictionaries
    mapping feature names to their respective feature values.
    """
    _left_context_size = 0
    _right_context_size = 0
    _minimum_left_context_size = 0
    _minimum_right_context_size = 0

    def __init__(self, sequenced=False):
        self._label_mapper = LabelMapper()
        self.sequenced = sequenced

    @property
    def context_size(self):
        """The maximum number of entries the feature extractor looks
        behind/ahead when extracting features from a sequence.

        Given as (left_context_size, right_context_size).
        """
        return (self._left_context_size, self._right_context_size)

    @context_size.setter
    def context_size(self, size):
        (left, right) = size
        if left < self._minimum_left_context_size:
            raise ValueError("left context size too small ({0} < {1})"\
                             .format(left, self._minimum_left_context_size))
        if right < self._minimum_right_context_size:
            raise ValueError("right context size too small ({0} < {1})"\
                             .format(right, self._minimum_right_context_size))
        self._left_context_size = left
        self._right_context_size = right

    @property
    def feature_count(self):
        """The currently known number of possible feature names."""
        return len(self._label_mapper)

    @property
    def features(self):
        return self._label_mapper

    @property
    def sequenced(self):
        return self._sequenced

    @sequenced.setter
    def sequenced(self, status):
        self._sequenced = status
        self._rebind_methods(status)

    def _rebind_methods(self, status):
        if status:
            self.init = self._init_sequenced
            self.get = self._get_sequenced
            self.get_vector = self._get_vector_sequenced
        else:
            self.init = self._init_independent
            self.get = self._get_independent
            self.get_vector = self._get_vector_independent

    def _init_independent(self, dataset):
        """Initialize the feature extractor with a dataset.

        Called when training on a new dataset, this function can be used to
        extract frequency/distributional information from the data to be used in
        features, or to pre-compute some features.
        """
        pass

    def _init_sequenced(self, dataset):
        """Initialize the feature extractor with a sequential dataset.
        """
        pass

    def _get_independent(self, x):
        """Return the feature representation for a given input."""
        raise NotImplementedError("function not implemented")

    def _get_sequenced(self, seq, pos, history=None):
        """Return the feature representation for a given data point in a
        sequence.
        """
        raise NotImplementedError("function not implemented")

    def _get_vector_independent(self, x):
        """Return a numerical feature vector for a given input."""
        return self._label_mapper.map_to_vector(self._get_independent(x))

    def _get_vector_sequenced(self, seq, pos, history=None):
        """Return a numerical feature vector for a given data point in a
        sequence.
        """
        return self._label_mapper.map_to_vector(
            self._get_sequenced(seq, pos, history=history))

    def __getstate__(self):
        return {
            'label_mapper': self._label_mapper,
            'context_size': self.context_size,
            'sequenced': self.sequenced
            }

    def __setstate__(self, state):
        self._label_mapper = state['label_mapper']
        self.context_size = state['context_size']
        self.sequenced = state['sequenced']
