#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import gzip
import pickle

def main():
    with gzip.open(args.parfile, 'rb') as f:
        model = pickle.load(f)
    model.print_weights()

if __name__ == '__main__':
    description = "Print weights from a trained perceptron."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('parfile',
                        metavar='PARFILE',
                        type=str,
                        help='Parameter file (required)')

    args = parser.parse_args()
    main()
