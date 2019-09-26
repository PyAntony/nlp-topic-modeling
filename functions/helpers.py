import time
import string
from functional import seq
from functools import reduce, partial
import stopwordsiso as stopwords
from typing import List

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer


def decorator_timeit(func):
    """Decorator to time a function processing time."""

    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        end = round((time.time() - start) / 60, 2)
        print(f'TIMEIT DECORATOR: {func.__name__} took: {end} min.')
        return results

    return wrapper


def readCorpus(file_path):
    """ Read file as json or csv. Return a DataFrame."""
    if '.json' in file_path:
        return pd.read_json(file_path, lines=True)
    else:
        return pd.read_csv(file_path)


def rawAnalyzer(nlp, stopWords: List[str], document: str):
    """ Analyzer for TfidfVectorizer (preprocessor + tokenizer)"""
    # dictionary to replace all punctuations
    strTable = str.maketrans('', '', string.punctuation)
    # lower and tokenize using spacy module
    tokens = nlp(document.lower())

    return (seq(tokens)
            # lemmatize each token
            .map(lambda token: token.lemma_)
            # remove emails
            .filter(lambda s: '@' not in s)
            # remove additional garbage (mostly for html documents)
            .filter_not(lambda s: any(v for v in ['www', 'http', 'https'] if v in s and len(s) > 5))
            # remove all punctuations and strip empty spaces
            .map(lambda s: s.translate(strTable).strip())
            # remove tokens with single letters
            .filter(lambda s: len(s) > 1)
            # remove tokens with combinations of letters and numbers
            .filter_not(lambda s: any(c.isdigit() for c in s))
            # remove stop words
            .filter(lambda s: s not in stopWords)
            )


def getStopWords(spacy_model):
    """Stop words tokenized with the default raw analyzer."""
    # for languages available go to: https://github.com/stopwords-iso
    s_words = stopwords.stopwords('en')

    analyzer = partial(rawAnalyzer, spacy_model, [])
    return seq(s_words).flat_map(analyzer).to_list()


def getVectorizer(vectorizerType, analyzer):
    """ Return a TfidfVectorizer instance."""
    if vectorizerType == 'lda':
        # 'use_idf' and 'norm' == False, equal to CountVectorizer
        return TfidfVectorizer(use_idf=False, norm=None, analyzer=analyzer)

    return TfidfVectorizer(analyzer=analyzer)


def prepareXyTrain(corpus: DataFrame, XtrainCol: str, y_train_col: str):
    """Returns the X_train matrix and the target column from the corpus."""
    Xcol = XtrainCol.split('*') if '*' in XtrainCol else XtrainCol

    if isinstance(Xcol, str):
        return corpus[Xcol], corpus[y_train_col]

    # else if multiple columns are concatenated (Xcol: List[String])
    strConcatenator = lambda row: reduce(lambda a, b: a + ' ' + b, row)
    Xtrain = corpus[Xcol].apply(strConcatenator, axis=1)
    return Xtrain, corpus[y_train_col]


def dfTopFeatures(features: List[str], headers: List[str], values: np.ndarray,
                  subHeaders: List[str], targetType: str, nTop: int = 20) -> DataFrame:
    """
    Construct a DataFrame with columns containing the sorted top features.

    Parameters
    ----------
    features: list of features.
    headers: list of DataFrame level 1 columns (e.g., target classes or topics).
    values: numerical values as odds or probabilities.
            Shape of 2d array: (n_headers, n_features).
    subHeaders: list of DataFrame level 2 columns (e.g., ['feature', 'odd']).
    targetType: to be used as the index of the level 1 columns(e.g., 'topic').
    nTop: top number of features to select.
    """
    rawPairs = (seq(values)
                .map(lambda sq: list(zip(features, sq)))
                .map(lambda sq: sorted(sq, reverse=True, key=lambda tp: tp[1]))
                .map(lambda sq: sq[:nTop])
                .to_list()
                )

    # prepare column labels (2 levels)
    iterables = [headers, subHeaders]
    index = pd.MultiIndex.from_product(iterables, names=[targetType, 'values'])
    # prepare numpy matrix with rawPairs
    matrix = pd.concat([pd.DataFrame(d) for d in rawPairs], axis=1).to_numpy()

    return pd.DataFrame(matrix, columns=index, index=range(1, nTop + 1))
