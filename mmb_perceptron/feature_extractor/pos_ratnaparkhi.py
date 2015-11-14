# -*- coding: utf-8 -*-

from collections import Counter
import itertools as it
from .feature_extractor import FeatureExtractor

class Ratnaparkhi(FeatureExtractor):
    """Feature vectors based on Ratnaparkhi (1996).

    Ratnaparkhi, A. (1996). A maximum-entropy model for part-of-speech tagging.
    """
    _minimum_left_context_size = 2
    _minimum_right_context_size = 1
    _wordfreq = Counter()
    _freq_threshold = 4

    def _init_sequenced(self, dataset):
        all_words = it.chain.from_iterable((words for words in dataset))
        self._wordfreq = Counter(all_words)

    def _get_sequenced(self, seq, pos, history=None):
        word = seq[pos]
        features = {}
        features[u'bias'] = 1.0
        features[u'left_tag ' + history[pos - 1]] = 1.0
        features[u'left_tag_bigram ' + history[pos - 2] + ' ' + history[pos - 1]] = 1.0
        if self._wordfreq[word] > self._freq_threshold:
            features[u'this_word ' + word] = 1.0
        else:
            for i in range(1, 5):
                features[u'this_prefix_{0} {1}'.format(i, word[:i])] = 1.0
            for i in range(-4, 0):
                features[u'this_suffix_{0} {1}'.format(i, word[i:])] = 1.0
            if any(c.isdigit() for c in word):
                features[u'has_number'] = 1.0
            if '-' in word:
                features[u'has_hyphen'] = 1.0
        for i in range(1, self._left_context_size + 1):
            features[u'left_{0}_word {1}'.format(i, seq[pos - i])] = 1.0
        for i in range(1, self._right_context_size + 1):
            features[u'right_{0}_word {1}'.format(i, seq[pos + i])] = 1.0
        return features
