#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import gzip
import itertools as it
import operator as op
import pickle
import sys
from mmb_perceptron.mixed_impl import CombinatorialPerceptron
from mmb_perceptron.feature_extractor import \
     Honnibal, Ratnaparkhi, Char
from mmb_perceptron.helper.pos_tagging import \
     log, check_counts_for_mode, extract_sentences, preprocess_sentences

def get_feature_extractor(name, context_size):
    if name == 'Honnibal':
        feature = Honnibal(sequenced=True)
    elif name == 'Ratnaparkhi':
        feature = Ratnaparkhi(sequenced=True)
    elif name == 'Char':
        feature = Char(sequenced=True)
    else:
        raise NotImplementedError(name + " is not implemented")
    feature.context_size = (context_size, context_size)
    return feature

def main():
    log("Reading input data...")
    (sentences, gold_tags, token_count, tag_count) = \
        extract_sentences(args.infile, encoding=args.enc)
    log("Parsed {0} token(s) with {1} tags in {2} sentence(s)."
        .format(token_count, tag_count, len(sentences)))
    check_counts_for_mode(token_count, tag_count, args.train)

    if args.preprocessing:
        sentences = preprocess_sentences(sentences)

    if args.train:
        log("Training...")
        model = CombinatorialPerceptron(
            averaged=args.averaging,
            iterations=args.iterations,
            learning_rate=1,
            sequenced=True,
            feature_extractor=get_feature_extractor(args.feature, args.context_size),
            log_to=sys.stderr
            )
        model.train(sentences, gold_tags)
        log("Saving...")
        with gzip.open(args.par, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    if not args.train:
        log("Loading model...")
        with gzip.open(args.par, 'rb') as f:
            model = pickle.load(f)
            model.log_to = sys.stderr
        log("Tagging...")
        correct_count = 0
        predictions = model.predict_all(sentences)
        for sentence in it.izip(sentences, predictions, gold_tags):
            for (word, pred_tag, gold_tag) in it.izip(*sentence):
                print(u"{0}\t{1}".format(word, pred_tag).encode("utf-8"))
                if gold_tag is not None and gold_tag == pred_tag:
                    correct_count += 1
            print() # line break between sentences
        if tag_count > 0:  # print evaluation
            log("Accuracy:  {0:7}/{1:7} correct ({2:.2f}%)"
                .format(correct_count, tag_count, (float(correct_count)/tag_count)*100))

    log("Done.")


if __name__ == '__main__':
    description = "Part-of-speech tagger using a greedy perceptron algorithm."
    epilog = ("Input is expected to contain one token per line, with an empty "
              "line being treated as a sentence boundary.  Tokens and "
              "gold-standard POS tags (required for training, optional for "
              "tagging) must be separated by a <tab>.")
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument('infile',
                        metavar='INPUT',
                        type=argparse.FileType('r'),
                        help='Input file (required)')
    parser.add_argument('-e', '--encoding',
                        dest='enc',
                        default='utf-8',
                        help='Encoding of the input file (default: '
                             '%(default)s)')
    parser.add_argument('-p', '--par',
                        metavar='PARFILE',
                        required=True,
                        type=str,
                        help='Parameter file (required)')
    parser.add_argument('-t', '--train',
                        action='store_true',
                        default=False,
                        help='Train on INPUT and save the resulting '
                             'parametrization to PARFILE')

    model_group = parser.add_argument_group('model parameters')
    model_group.add_argument('-c', '--context-size',
                             metavar='N',
                             dest='context_size',
                             type=int,
                             default=2,
                             help='Number of tokens to use as context '
                             '(default: %(default)i)')
    model_group.add_argument('-f', '--feature',
                             choices=('Honnibal', 'Ratnaparkhi', 'Char'),
                             default='Ratnaparkhi',
                             help='Feature extractor (default: %(default)s)')
    model_group.add_argument('-i', '--iterations',
                             metavar='N',
                             type=int,
                             default=5,
                             help='Number of iterations for training '
                             '(default: %(default)i)')
    model_group.add_argument('--no-averaging',
                             dest='averaging',
                             action='store_false',
                             default=True,
                             help='Don\'t average the model parameters')
    model_group.add_argument('--no-preprocessing',
                             dest='preprocessing',
                             action='store_false',
                             default=True,
                             help=('Don\'t perform preprocessing on tokens '
                                   '(useful if you want to do your own '
                                   'preprocessing, or use this tagger for some '
                                   'other labelling problem)'))

    args = parser.parse_args()
    main()
