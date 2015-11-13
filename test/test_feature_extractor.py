# -*- coding: utf-8 -*-

import pytest
from mmb_perceptron.feature_extractor import FeatureExtractor

class TestFeatureExtractor(object):
    """Tests for feature extractors.
    """

    def test_context_size(self):
        f = FeatureExtractor()
        assert f.context_size == (0, 0)
        f.context_size = (1, 2)
        assert f.context_size == (1, 2)
        with pytest.raises(ValueError):
            f.context_size = (-1, 1)
        with pytest.raises(ValueError):
            f.context_size = (1, -1)
        assert f.context_size == (1, 2)
