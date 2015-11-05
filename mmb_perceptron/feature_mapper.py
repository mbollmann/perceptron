# -*- coding: utf-8 -*-

import collections

class FeatureMapper(collections.Mapping):
    """Maps arbitrary (feature) names to vector indices.

    Behaves mostly like a read-only dictionary: accessing an element with
    bracket notation will return a vector index for that element; if it doesn't
    exist yet, it will be created automatically.
    """
    def __init__(self):
        self.features = {}

    def __getitem__(self, key):
        try:
            value = self.features[key]
        except KeyError:
            value = self.add(key)
        return value

    def __contains__(self, key):
        return key in self.features

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return self.features.__iter__()

    def get(self, key, default=None):
        return self.features.get(key, default)

    def add(self, item):
        value = self.features[item] = len(features)
        return value

    def map_list(self, elems):
        """Maps a list of labels and returns a list of indices."""
        return [self[e] for e in elems]
