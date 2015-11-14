# -*- coding: utf-8 -*-

import sys

class InputFormatError(Exception):
    pass

def log(text, type="info"):
    if type == 'warn': sys.stderr.write("\033[1;33m")
    if type == 'error': sys.stderr.write("\033[1;31m")
    if type == 'info!': sys.stderr.write("\033[1;32m")
    sys.stderr.write("[{:^5}] ".format(type))
    try:
        sys.stderr.write(text)
    except UnicodeError:
        sys.stderr.write(text.encode(encoding))
    sys.stderr.write("\033[0m\n")

def check_counts_for_mode(token_count, tag_count, train_mode):
    """Makes sure that all tokens have tags when using training mode,
       and warns if only some of the tokens have tags when evaluating."""
    if token_count != tag_count:
        if train_mode:
            raise InputFormatError("Each token must have a tag when training")
        elif tag_count > 0:
            log("Only some of the tokens have a gold tag! "
                "Evaluation will be inaccurate.", type="warn")

def extract_sentences(data, encoding="utf-8"):
    """Parses input into sentence arrays suitable for the tagger."""
    words, tags = [], []
    word_sequences, tag_sequences = [], []
    token_count, tag_count = 0, 0
    for i, line in enumerate(data):
        line = line.strip().decode(encoding)
        if line:
            if line.count("\t") == 0:
                words.append(line)
                tags.append(None)
            elif line.count("\t") == 1:
                (word, tag) = line.split("\t")
                words.append(word)
                tags.append(tag)
                tag_count += 1
            else:
                raise InputFormatError(
                    "Line {0} has an incorrect number of tabs: {1}"
                    .format(i, line)
                    )
        elif words: # empty line treated as "end-of-sentence"
            word_sequences.append(words)
            tag_sequences.append(tags)
            token_count += len(words)
            words, tags = [], []

    if words:
        word_sequences.append(words)
        tag_sequences.append(tags)
        token_count += len(words)

    if not word_sequences or token_count == 0:
        raise InputFormatError("No input data")

    return (word_sequences, tag_sequences, token_count, tag_count)

def preprocess_word(word):
    """Preprocess a word for POS tagging.

    Currently does not do much except lowercase everything and convert tokens
    starting with a digit to a special '__DIGIT__' token.
    """
    if word[0].isdigit():
        return '__DIGIT__'
    else:
        return word.lower()

def preprocess_sentences(sentences):
    """Preprocess a list of sentences for tagging.
    """

    processed = []
    for words in sentences:
        processed.append([preprocess_word(word) for word in words])
    return processed
