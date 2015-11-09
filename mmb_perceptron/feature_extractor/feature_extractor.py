# -*- coding: utf-8 -*-

from ..label_mapper import LabelMapper

class FeatureExtractor(object):
    """Abstract base class for a feature extractor.

    Feature extractors take (potentially arbitrary) data points as input and
    convert them into feature representations, in the form of dictionaries
    mapping feature names to their respective feature values.
    """
    _context_size = (0, 0)

    def __init__(self, sequenced=False):
        self._label_mapper = LabelMapper()
        self.sequenced = sequenced

    @property
    def context_size(self):
        """The maximum number of entries the feature extractor looks
        behind/ahead when extracting features from a sequence.

        Given as (left_context_size, right_context_size).
        """
        return self._context_size

    @context_size.setter
    def context_size(self, left, right):
        assert left >= 0
        assert right >= 0
        self._context_size = (left, right)

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
        if status:
            self.init = self._init_independent
            self.get = self._get_sequenced
            self.get_vector = self._get_vector_sequenced
        else:
            self.init = self._init_sequenced
            self.get = self._get_independent
            self.get_vector = self._get_vector_independent

    def _init_independent(self, dataset):
        """Initialize the feature extractor with a dataset.

        Called when training on a new dataset, this function can be used to
        extract frequency/distributional information from the data to be used in
        features, or to pre-compute some features.
        """
        raise NotImplementedError("function not implemented")

    def _init_sequenced(self, dataset):
        """Initialize the feature extractor with a sequential dataset.
        """
        raise NotImplementedError("function not implemented")

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
