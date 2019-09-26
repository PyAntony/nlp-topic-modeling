import numpy as np
from pandas import Series

import os
import glob
import time
import pickle
from typing import List

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from functions.helpers import readCorpus
from functions.helpers import decorator_timeit
from functions.helpers import prepareXyTrain
from functions.helpers import getVectorizer
from functions.helpers import dfTopFeatures


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

    O = P(Xi) / 1 - P(Xi) = e^(B^t * X) , where O is the odds ratio.
    log(O) = log(e^(B^t * X)) = B^t * X = logits
    We optimize B^t to minimize the cost (resembles a perceptron).

    Odd ratios:
    ----------
    It is the % increase for 1 unit; e.g., 1.30 represents 30% increase for P(xi).
    """
    # mapping from feature integer indices to feature name (e.g., words)
    features: List[str] = fittedPipe.named_steps['tfIdf'].get_feature_names()
    # list of classes
    classes: List[str] = list(fittedPipe.classes_)
    # size = (n_targets, n_features)
    coefficients: np.ndarray = fittedPipe.named_steps['logModel'].coef_

    rCoeffs = np.apply_along_axis(lambda n: np.round(n, 2), 1, coefficients)
    l2titles = ['feature', 'odd']
    return dfTopFeatures(features, classes, rCoeffs, l2titles, 'target', nFeatures)


def modelSaver(fittedPipe, supportDf, coeffsDf):
    """Save fitted models as a pickle file and dataframes as csv files."""
    today = time.strftime('%Y%m%d')
    path = 'output/logistic/'
    if not os.path.exists(path):
        os.makedirs(path)

    # saving files
    pickle.dump(fittedPipe, open(f'{path}fitted_pipe{today}.sav', 'wb'))
    supportDf.to_csv(f'{path}support{today}.csv')
    coeffsDf.to_csv(f'{path}features{today}.csv')

    print('files saved successfully.')


def logistic_main(analyzer, x_train_col, y_train_col, n_features):
    """ Logistic Regression entry function."""
    # Pipeline to process corpus
    pipeline = Pipeline([
        ('tfIdf', getVectorizer('logistic', analyzer)),
        ('logModel', LogisticRegression(solver='lbfgs', multi_class='auto', n_jobs=-1))
    ])

    inputFile = glob.glob('input/*')[0]
    corpus = readCorpus(inputFile)

    # prepare X and y train sets
    X_train, y_train = prepareXyTrain(corpus, x_train_col, y_train_col)

    # produce a fitting function to timeit
    @decorator_timeit
    def fitter(pipe, X, y):
        return pipe.fit(X, y)

    print('fitting models...')
    fittedPipe = fitter(pipeline, X_train, y_train)

    print('preparing support and coefficients...')
    supportDf = getSupport(y_train)
    coeffsDf = getCoeffMatrix(fittedPipe, n_features)

    print("\n=== TOP FEATURES ===")
    print(coeffsDf.head(n_features))
    print("\n=== SUPPORT ===")
    print(supportDf.head(supportDf.shape[0]))
    print()

    # save trained model and dataframes
    modelSaver(fittedPipe, supportDf, coeffsDf)
