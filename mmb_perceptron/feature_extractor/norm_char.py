# -*- coding: utf-8 -*-

from .feature_extractor import FeatureExtractor

class Char(FeatureExtractor):
    """Feature vectors for normalization as character-based tagging.
    """

    def _get_sequenced(self, seq, pos, history=None):
        joined = ''.join(history).replace("__EPS__", "")
        features = {}
        features[u'bias'] = 1.0
        features[u'this_char ' + seq[pos]] = 1.0
        full_word = seq[(self._left_context_size + 1):-self._right_context_size]
        features[u'full_word ' + ''.join(full_word)] = 1.0
        for i in range(1, self._left_context_size + 1):
            features[u'left_{0}_char {1}'.format(i, seq[pos - i])] = 1.0
            features[u'left_upto_{0}_tags {1}'\
                     .format(i, ' '.join(history[(pos - i):pos]))] = 1.0
            features[u'left_joined_{0}_tags {1}'.format(i, joined[-i:])] = 1.0
        for i in range(1, self._right_context_size + 1):
            features[u'right_{0}_char {1}'.format(i, seq[pos + i])] = 1.0
        return features
