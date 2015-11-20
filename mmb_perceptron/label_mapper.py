# -*- coding: utf-8 -*-

import collections
import numpy as np

class LabelMapper(collections.Mapping):
    """Maps arbitrary (feature/class) labels to vector indices.

    Behaves mostly like a read-only dictionary: accessing an element with
    bracket notation will return a vector index for that element; if it doesn't
    exist yet, it will be created automatically.
    """

    def __init__(self):
        self.features = {}
        self.featurelist = []

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

    def get_name(self, value):
        return self.featurelist[value]

    def get_names(self, values):
        return [self.featurelist[v] for v in values]

    def add(self, item):
        value = self.features[item] = len(self.features)
        self.featurelist.append(item)
        return value

    def extend(self, elems):
        for e in elems:
            if e not in self.features:
                self.add(e)

    def map_list(self, elems):
        """Maps a list of labels and returns a list of indices."""
        return [self[e] for e in elems]

    def map_to_vector(self, feat):
        self.extend(feat)
        vec = np.zeros(len(self.features))
        for name, value in feat.iteritems():
            vec[self[name]] = value
        return vec

    def get_vector(self, feat):
        vec = np.zeros(len(self.features))
        for name, value in feat.iteritems():
            if name in self.features:
                vec[self[name]] = value
        return vec

    def prune_indices(self, indices):
        """Prune the given indices.

        Deletes the associated labels and remaps all indices.
        """
        for idx in sorted(indices, reverse=True):
            del self.featurelist[idx]
        self.features = {f: idx for idx, f in enumerate(self.featurelist)}

    def reset(self):
        self.__init__()
