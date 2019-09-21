import numpy as np
import pandas as pd

import time
import os
import math
import pickle
from functools import reduce
from typing import List

import stopwordsiso as stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from pandas import DataFrame, Series


def decorator_timeit(func):
    """Decorator to time a function processing time."""
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        end = round((time.time() - start)/60, 2)
        print(f'TIMEIT DECORATOR: {func.__name__} took: {end} min.')
        return results

    return wrapper


def getStopWords(language: str):
    """Stop words tokenized with the default TfidfVectorizer analyzer."""
    sWords = stopwords.stopwords(language)
    defaultAnalyzer = TfidfVectorizer(analyzer='word').build_analyzer()

    return set([s for ls in [defaultAnalyzer(w) for w in sWords] for s in ls])


def prepareXyTrain(corpus: DataFrame, XtrainCol: str, y_train_col: str):
    """Returns the X_train matrix and the target column from the corpus."""
    Xcol = XtrainCol.split('*') if '*' in XtrainCol else XtrainCol

    if isinstance(Xcol, str):
        return corpus[Xcol], corpus[y_train_col]

    # else if multiple columns are concatenated (Xcol: List[String])
    strConcatenator = lambda row: reduce(lambda a, b: a + ' ' + b, row)
    Xtrain = corpus[Xcol].apply(strConcatenator, axis=1)
    return Xtrain, corpus[y_train_col]


def getSupport(y: Series):
    """Count classes in y_train to get the support."""
    ys = y.value_counts().sort_values(ascending=False).reset_index()
    ys.columns = ['yTarget', 'count']
    return ys


def getCoeffMatrix(fittedPipe: Pipeline, nFeatures: int):
    """
    Return a Dataframe that contains the top 'nFeatures' (features with
    highest odds), 1 column by target dependent variable.

    Activation function (logistic function):
    ----------------------------------------
    P(xi) = e^(B^t * X) / (1 + e^(B^t * X)),
        where P(Xi) is the probability of the positive label.
    For multi-class we have SOFTMAX:
    P(xi) = e^(B^t * X) / SUM(e^(B^ti * X))
        
    O = P(Xi) / 1 - P(Xi), where O is the odds ratio.
    O = e^(B^t * X), by simplification.

    Notice that the coefficients (logits) are the log of the odds:
    log(O) = B^t * X
    We want to optimize B^t in order to minimize the cost.

    Odd ratios:
    ----------
    It is the % increase for 1 unit; e.g., 1.30 represents 30% increase for P(xi).
    """
    # mapping from feature integer indices to feature name (e.g., words)
    features: List[str] = fittedPipe.named_steps['tfIdf'].get_feature_names()
    # list of classes
    classNames: np.ndarray = fittedPipe.classes_
    # size = (n_targets, n_features)
    coefficients: np.ndarray = fittedPipe.named_steps['logModel'].coef_

    # zip odds with features, sort by odd, and get top 'nFeatures'
    def getTopFeatures(oddsList):
        zippWithFeatures = lambda oddsSeq: list(zip(features, oddsSeq))
        sorting = lambda ls: sorted(ls, reverse=True, key=lambda tp: tp[1])
        return sorting(zippWithFeatures(oddsList))[:nFeatures]

    # logits to odds
    toOdd = lambda n: np.round(math.e**n, 2)
    odds = np.apply_along_axis(toOdd, 1, coefficients)

    # prepare matrix with features and odds as columns
    data = [getTopFeatures(ls) for ls in odds]
    matrix = pd.concat([pd.DataFrame(d) for d in data], axis=1).to_numpy()
    # dataframe multiIndex columns
    iterables = [classNames, ['feature', 'odd']]
    index = pd.MultiIndex.from_product(iterables, names=['target', 'values'])

    return pd.DataFrame(matrix, columns=index)


def modelSaver(dirSaveName, fittedPipe, supportDf, coeffsDf):
    """Save fitted models as a pickle file and dataframes as csv files."""
    today = time.strftime('%Y%m%d')
    path = f'trained_models/{dirSaveName}'

    # overwrites existing files if processed in the same day
    os.makedirs(path) if not os.path.exists(path)\
        else print('folder exists. Saving files in the same folder...')

    # saving files
    pickle.dump(fittedPipe, open(f'{path}/fitted_pipe{today}.sav', 'wb'))
    supportDf.to_csv(f'{path}/support{today}.csv')
    coeffsDf.to_csv(f'{path}/features{today}.csv')

    print('files saved successfully.')
