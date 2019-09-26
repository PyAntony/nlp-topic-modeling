import glob
import os
from typing import List, Tuple, Callable
from functools import partial

import seaborn as sns
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

from functions.helpers import dfTopFeatures, decorator_timeit
from functions.helpers import readCorpus
from functions.helpers import prepareXyTrain
from functions.helpers import getVectorizer


def createOutputDirectory(iPath, n_topics):
    """ Create output directory from input file name."""
    inputFileName = iPath.split('/')[-1].split('.')[0]
    oDir = f'output/{inputFileName}/topics_{n_topics}/'

    if not os.path.exists(oDir):
        os.makedirs(oDir)

    return oDir


def prepareDfs(dfPercent):
    """
    Prepare 2 DataFrames:
    - dfCount has count of all documents per topic (topic frequency).
    It includes all articles that mention the topic (even if only %1).
    It also gets the probability distribution given those frequencies.
    - dfNorm count all topic probabilities across all documents and then
    normalizes all the probabilities. It is a more exact measurement of the
    proportion of the topic coverage independent of the number of documents
    talking about it.
    """
    colNames = {'index': 'topic', 0: 'count'}

    # DF with row count if topic is present
    dfBinary = dfPercent.applymap(lambda n: 0 if n < 0.01 else 1)
    dfCount = dfBinary.sum().reset_index().rename(columns=colNames)
    dfCount['perct'] = dfCount['count'] / dfCount['count'].sum()

    # DF with added percentages normalized across all documents
    total = dfPercent.sum().sum()
    dfNorm = (dfPercent.sum() / total).reset_index().rename(columns=colNames)

    return dfCount, dfNorm


def prepareBarPLots(dfCount, dfNorm, saveDir):
    """ Prepare titles and y labels for 3 barplots."""
    title = 'Count of Documents with Topic'
    yL = 'count ( > 0% )'
    plotBars('topic', 'count', dfCount, title, yL, saveDir + 'count.png')

    title = 'Percentage of Documents with Topic'
    yL = 'percentage'
    plotBars('topic', 'perct', dfCount, title, yL, saveDir + 'percent1.png')

    title = 'Total % Across Documents (normalized)'
    plotBars('topic', 'count', dfNorm, title, yL, saveDir + 'percent2.png')


def plotBars(x, y, data, title, ylabel, savePathName):
    """ Create and save barplot."""
    sns.barplot(x=x, y=y, data=data, palette='YlGn')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(ls='--', alpha=.6)

    plt.savefig(savePathName)
    plt.close()


def plotLine(yAxis, data, color, savePathName):
    """ Produce and save lineplots for LDA scores."""
    sns.lineplot(x='k', y=yAxis, data=data, linewidth=4, color=color)
    plt.title(f"{yAxis.capitalize()} vs. Number of Topics")
    plt.xlabel('components')
    plt.grid(ls='--', alpha=.6)

    plt.savefig(savePathName)
    plt.close()


@decorator_timeit
def computeLDA(analyzer, xCol: str, nWords: int, n_topics: int, file: str):
    """ Compute LDA process for 1 file."""
    print(f'processing LDA for {file} and [{n_topics}] topics...')

    # create output directory to store results
    outputDir = createOutputDirectory(file, n_topics)
    rawCorpus = readCorpus(file)
    # get X_train. y_train is ignored
    X_train = prepareXyTrain(rawCorpus, xCol, rawCorpus.columns[0])[0]

    tfidf = getVectorizer('lda', analyzer)
    sparseX = tfidf.fit_transform(X_train)
    # lda model with default parameters
    lda = LatentDirichletAllocation(n_components=n_topics, n_jobs=-1)
    # size(n_documents, n_topics). Data = topic probas
    dfVectorized = pd.DataFrame(lda.fit_transform(sparseX))

    # prepare 2 dataframes to visualize frequencies and percentages
    dfCounts, dfProbasNormalized = prepareDfs(dfVectorized)
    prepareBarPLots(dfCounts, dfProbasNormalized, outputDir)

    # normalize and round lda components (size is (n_topics, n_features))
    probas = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    rProbas = np.apply_along_axis(lambda n: np.round(n, 4), 1, probas)

    # produce and save dataframe with top features
    features = tfidf.get_feature_names()
    headers = ["topic_" + str(c) for c in dfVectorized.columns]
    l2headers = ['word', 'proba']
    topDf = dfTopFeatures(features, headers, rProbas, l2headers, 'topic', nWords)
    topDf.to_csv(f'{outputDir}topFeatures.csv')

    return [lda.score(sparseX), lda.perplexity(sparseX)]


def getLdaScoresDf(funcLDA: Callable, kfPairs: List[Tuple], acc=None) -> DataFrame:
    """
    Compute an LDA process for each (n_components, file) pair
    and return a DataFrame with the models scores.
    """
    if acc is None:
        acc = list()
    if not kfPairs:
        cols = ['k', 'file', 'likelihood', 'perplexity']
        return pd.DataFrame(acc, columns=cols)

    nComponents, inputFile = kfPairs[0]
    scores = funcLDA(nComponents, inputFile)

    data = [nComponents, inputFile, scores[0], scores[1]]
    return getLdaScoresDf(funcLDA, kfPairs[1:], acc + [data])


def getInputPairs(lda_components):
    """ Produce (n_components, file) pairs."""
    try:
        inputPaths = glob.glob("input/*")
        k0 = int(lda_components.split('-')[0])
        k1 = int(lda_components.split('-')[1])
        return [(k, file) for k in range(k0, k1 + 1) for file in inputPaths]
    except:
        return 'error'


@decorator_timeit
def lda_main(analyzer, lda_components, x_train_col, n_words):
    """ LDA process entry function."""
    kfPairs = getInputPairs(lda_components)

    if 'error' in kfPairs:
        print('LDA Error: unable to generate input files.')
        return

    # curry LDA function for single process
    computeLDA_ = partial(computeLDA, analyzer, x_train_col, n_words)
    # DataFrame with likelihoods and perplexities for all files and k topics
    finalDf = getLdaScoresDf(computeLDA_, kfPairs)

    def getDir(s):
        return f"output/{s.split('/')[-1].split('.')[0]}/"

    finalDf['outDir'] = finalDf['file'].map(getDir)

    # save lineplots for each score for each file
    for outDir in set(finalDf['outDir']):
        df = finalDf.loc[finalDf['outDir'] == outDir].sort_values('k')

        plotLine('likelihood', df, '#87CEFA', outDir + 'likelihood.png')
        plotLine('perplexity', df, '#FFA500', outDir + 'perplexity.png')
