#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import gzip
import itertools as it
import operator as op
import pickle
import progressbar as pb
import sys
from mmb_perceptron.dict_impl import \
     CombinatorialPerceptron as CombinatorialPerceptron_Dict
from mmb_perceptron.mixed_impl import \
     CombinatorialPerceptron, CombinatorialViterbiPerceptron
from mmb_perceptron.numpy_impl import \
     CombinatorialPerceptron as CombinatorialPerceptron_Numpy
from mmb_perceptron.feature_extractor import \
     Honnibal, Ratnaparkhi, Char
from mmb_perceptron.helper.pos_tagging import \
     Logger, check_counts_for_mode, extract_sentences, preprocess_sentences

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

def get_perceptron_model(impl, structured):
    if structured:
        return CombinatorialViterbiPerceptron
    if impl == 'dict':
        return CombinatorialPerceptron_Dict
    elif impl == 'numpy':
        return CombinatorialPerceptron_Numpy
    return CombinatorialPerceptron

def main():
    Logger.log("Reading input data...")
    (sentences, gold_tags, token_count, tag_count) = \
        extract_sentences(args.infile, encoding=args.enc)
    Logger.log("Parsed {0} token(s) with {1} tags in {2} sentence(s)."
        .format(token_count, tag_count, len(sentences)))
    check_counts_for_mode(token_count, tag_count, args.train)

    if args.preprocessing:
        sentences = preprocess_sentences(sentences)

    if args.train:
        Logger.log("Training...")
        perceptron_model = get_perceptron_model(args.implementation, args.structured)
        model = perceptron_model(
            averaged=args.averaging,
            iterations=args.iterations,
            learning_rate=1,
            sequenced=True,
            feature_extractor=get_feature_extractor(args.feature, args.context_size)
            )
        widgets = [pb.Percentage(), ' ', pb.Bar(marker='#', left='[', right=']'), ' ',
                   pb.ETA(), '   ', pb.AbsoluteETA()]
        with pb.ProgressBar(max_value=(token_count * args.iterations),
                            redirect_stderr=True,
                            widgets=widgets) as bar:
            model.log_to = Logger
            model.progress_func = bar.update
            model.train(sentences, gold_tags)
        Logger.log("Saving...")
        with gzip.open(args.par, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    if not args.train:
        Logger.log("Loading model...")
        with gzip.open(args.par, 'rb') as f:
            model = pickle.load(f)
            model.log_to = Logger
        Logger.log("Tagging...")
        correct_count = 0
        nbest = (args.nbest > 1)
        if nbest:
            predictions = model.predict_all_nbest(sentences, n=args.nbest)
        else:
            predictions = model.predict_all(sentences)
        for sentence in it.izip(sentences, predictions, gold_tags):
            for (word, pred_tag, gold_tag) in it.izip(*sentence):
                if nbest:
                    print(u"{0}\t{1}".format(word, u"\t".join(pred_tag)).encode("utf-8"))
                    pred_tag = pred_tag[0]
                else:
                    print(u"{0}\t{1}".format(word, pred_tag).encode("utf-8"))
                if gold_tag is not None and gold_tag == pred_tag:
                    correct_count += 1
            print('') # line break between sentences
        if tag_count > 0:  # print evaluation
            Logger.log("Accuracy:  {0:7}/{1:7} correct ({2:.2f}%)"
                       .format(correct_count, tag_count,
                               (float(correct_count)/tag_count)*100))

    Logger.log("Done.")


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
    parser.add_argument('-n', '--nbest',
                        type=int,
                        default=1,
                        help=('Output the n-best suggestions for each token '
                              '(ignored during training)'))

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
    model_group.add_argument('--structured',
                             action='store_true',
                             default=False,
                             help=('Use structured prediction with Viterbi '
                                   'algorithm (CAUTION: extremely slow!)'))

    internal_group = parser.add_argument_group('internal parameters')
    internal_group.add_argument('--implementation',
                                choices=('dict','mixed','numpy'),
                                default='mixed',
                                help=('Choose perceptron implementation to use '
                                      '(should only affect computation speed) '
                                      '(default: %(default)s)'))

    args = parser.parse_args()
    main()
