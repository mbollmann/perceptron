# -*- coding: utf-8 -*-

from .feature_extractor import FeatureExtractor

class Honnibal(FeatureExtractor):
    """Feature extractor based on the POS tagger by Matthew Honnibal.

    <https://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/>
    """
    _minimum_left_context_size = 1
    _minimum_right_context_size = 1

    def _get_sequenced(self, seq, pos, history=None):
        word = seq[pos]
        features = {}
        features[u'bias'] = 1.0
        features[u'this_word ' + word] = 1.0
        features[u'this_suffix ' + word[-3:]] = 1.0
        features[u'this_prefix ' + word[0]] = 1.0
        features[u'left_suffix ' + seq[pos - 1][-3:]] = 1.0
        features[u'right_suffix ' + seq[pos + 1][-3:]] = 1.0
        for i in range(1, self._left_context_size + 1):
            features[u'left_{0}_tag {1}'.format(i, history[pos - i])] = 1.0
            features[u'left_{0}_word {1}'.format(i, seq[pos - i])] = 1.0
            if i == 1:
                features[u'this_word_left_tag ' + word + history[pos - i]] = 1.0
            else:
                features[u'left_upto_{0}_tags {1}'\
                         .format(i, ' '.join(history[(pos - i):pos]))] = 1.0
        for i in range(1, self._right_context_size + 1):
            features[u'right_{0}_word {1}'.format(i, seq[pos + i])] = 1.0
        return features
