# -*- coding: utf-8 -*-

class AbstractFeatureExtractor(object):
    """Abstract base class for a feature extractor.

    Feature extractors take (potentially arbitrary) data points as input and
    convert them into feature representations, in the form of dictionaries
    mapping feature names to their respective feature values.
    """
    _feature_set = ('bias',)

    def init(self, dataset):
        """Initialize the feature extractor with a dataset.

        Called when training on a new dataset, this function can be used to
        extract frequency/distributional information from the data to be used in
        features, or to pre-compute some features.
        """
        return self

    def get(self, x):
        """Return the feature representation for a given input."""
        return {'bias': 1.0}

    @property
    def feature_count(self):
        """Return the currently known number of possible feature names."""
        return len(self._feature_set)

    @property
    def features(self):
        return self._feature_set
