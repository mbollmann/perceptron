# -*- coding: utf-8 -*-

from ..label_mapper import LabelMapper

class FeatureExtractor(object):
    """Base class for a feature extractor.

    Feature extractors take (potentially arbitrary) data points as input and
    convert them into feature representations, in the form of dictionaries
    mapping feature names to their respective feature values.

    This feature extractor only returns a bias as its single feature.
    """
    _context_size = (0, 0)

    def __init__(self):
        self._label_mapper = LabelMapper()

    def init(self, dataset):
        """Initialize the feature extractor with a dataset.

        Called when training on a new dataset, this function can be used to
        extract frequency/distributional information from the data to be used in
        features, or to pre-compute some features.
        """
        self._label_mapper.add("bias")

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

    def get(self, x):
        """Return the feature representation for a given input."""
        return {'bias': 1.0}

    def get_seq(self, seq, pos, history=None):
        """Return the feature representation for a given data point in a
        sequence.
        """
        return self.get(seq[pos])

    def get_vector(self, x):
        """Return a numerical feature vector for a given input."""
        return self._label_mapper.map_to_vector(self.get(x))

    def get_vector_seq(self, seq, pos, history=None):
        """Return a numerical feature vector for a given data point in a
        sequence.
        """
        return self._label_mapper.map_to_vector(
            self.get_seq(seq, pos, history=history))
