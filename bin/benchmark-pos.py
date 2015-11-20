#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import itertools as it
import numpy as np
import random
import timeit

from mmb_perceptron.dict_impl import \
     CombinatorialPerceptron as CombinatorialPerceptron_Dict
from mmb_perceptron.numpy_impl import \
     CombinatorialPerceptron as CombinatorialPerceptron_Numpy
from mmb_perceptron.mixed_impl import \
     CombinatorialPerceptron as CombinatorialPerceptron_Mixed, \
     CombinatorialViterbiPerceptron
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

def make_models(modellist):
    models = []
    for m in modellist:
        if m == 'dict':
            models.append(CombinatorialPerceptron_Dict)
        elif m == 'numpy':
            models.append(CombinatorialPerceptron_Numpy)
        elif m == 'mixed':
            models.append(CombinatorialPerceptron_Mixed)
        elif m == 'viterbi':
            models.append(CombinatorialViterbiPerceptron)
    return models

def make_cv_splits(sentences, gold_tags, folds):
    split_sentences = []
    split_gold_tags = []
    start, stop = 0, 0
    for i in range(folds):
        stop = start + len(sentences[i::folds])
        split_sentences.append(sentences[start:stop])
        split_gold_tags.append(gold_tags[start:stop])
        start = stop
    for i in range(folds):
        t_s = [e for (n, s) in enumerate(split_sentences) for e in s if n != i]
        t_g = [e for (n, s) in enumerate(split_gold_tags) for e in s if n != i]
        training_split = (t_s, t_g)
        eval_split = (split_sentences[i], split_gold_tags[i])
        yield (training_split, eval_split)

def main():
    Logger.log("Reading input data...")
    (sentences, gold_tags, token_count, tag_count) = \
        extract_sentences(args.infile, encoding=args.enc)
    Logger.log("Parsed {0} token(s) with {1} tags in {2} sentence(s)."
        .format(token_count, tag_count, len(sentences)))
    check_counts_for_mode(token_count, tag_count, True)
    sentences = preprocess_sentences(sentences)

    models = make_models(args.models)
    all_splits = list(make_cv_splits(sentences, gold_tags, args.folds))

    for model in models:
        Logger.log("Benchmarking model " + model.__name__)
        accuracies = []
        times_train = []
        times_tag = []

        for n, (training_data, eval_data) in enumerate(all_splits):
            Logger.log("Processing fold {0}...".format(n+1))
            p = model(
                averaged=args.averaging,
                iterations=args.iterations,
                learning_rate=1,
                sequenced=True,
                feature_extractor=get_feature_extractor(args.feature, args.context_size),
                log_to=None
                )
            # training
            time_start = timeit.default_timer()
            p.train(*training_data)
            time_train = timeit.default_timer()

            # tagging
            (eval_sentences, eval_tags) = eval_data
            correct, eval_count = 0, 0
            predictions = p.predict_all(eval_sentences)
            time_tag = timeit.default_timer()

            # evaluating
            for sent in it.izip(predictions, eval_tags):
                correct += sum((guess == truth for guess, truth in it.izip(*sent)))
                eval_count += len(sent[0])
            accuracy = 1.0 * correct / eval_count
            delta_train = time_train - time_start
            delta_tag = time_tag - time_train

            Logger.log("  fold {0}: accuracy {1:.4f}, training time {2:.4f}, tagging time {3:.4f}"\
                .format(n+1, accuracy, delta_train, delta_tag))
            accuracies.append(accuracy)
            times_train.append(delta_train)
            times_tag.append(delta_tag)

        accuracies = np.array(accuracies)
        times_train = np.array(times_train)
        times_tag = np.array(times_tag)
        Logger.log("Evaluation results of model " + model.__name__, type="info!")
        Logger.log("       avg accuracy: {0:2.4f}   std: {1:.4f}"\
            .format(np.mean(accuracies), np.std(accuracies)), type="info!")
        Logger.log("  avg training time: {0:2.2f}     std: {1:2.2f}"\
            .format(np.mean(times_train), np.std(times_train)), type="info!")
        Logger.log("   avg tagging time: {0:2.2f}     std: {1:2.2f}"\
            .format(np.mean(times_tag), np.std(times_tag)), type="info!")

    Logger.log("Done.")

if __name__ == '__main__':
    description = "Benchmark for part-of-speech tagging with perceptron models."
    epilog = ""
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
    parser.add_argument('-o', '--folds',
                        metavar='N',
                        type=int,
                        default=10,
                        help='Number of folds for cross-validation '
                        '(default: %(default)i)')
    parser.add_argument('-m', '--models',
                        nargs='+',
                        choices=('dict','numpy','mixed','viterbi'),
                        default=['dict','numpy','mixed'],
                        help='Models to include in the benchmark')

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

    args = parser.parse_args()
    main()
