# -*- coding: utf-8 -*-

from ..label_mapper import LabelMapper

class FeatureExtractor(object):
    """Base class for a feature extractor.

    Feature extractors take (potentially arbitrary) data points as input and
    convert them into feature representations, in the form of dictionaries
    mapping feature names to their respective feature values.

    This feature extractor only returns a bias as its single feature.
    """
    def __init__(self):
        self._label_mapper = LabelMapper()

    def init(self, dataset):
        """Initialize the feature extractor with a dataset.

        Called when training on a new dataset, this function can be used to
        extract frequency/distributional information from the data to be used in
        features, or to pre-compute some features.
        """
        self._label_mapper.add("bias")

    def get(self, x):
        """Return the feature representation for a given input."""
        return {'bias': 1.0}

    def get_vector(self, x):
        """Return a numerical feature vector for a given input."""
        return self._label_mapper.map_to_vector(self.get(x))

    @property
    def feature_count(self):
        """Return the currently known number of possible feature names."""
        return len(self._label_mapper)

    @property
    def features(self):
        return self._label_mapper
