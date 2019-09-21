import click

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from functions.helpers import getStopWords
from functions.helpers import decorator_timeit
from functions.helpers import getSupport
from functions.helpers import getCoeffMatrix
from functions.helpers import prepareXyTrain
from functions.helpers import modelSaver


@click.command()
@click.option('--file_path', default='data.csv', show_default=True,
              help='Path of csv or json corpus file.')
@click.option('--X_train_col', default='X_train', show_default=True,
              help="X_train column name. Multiple column names can be passed with '*' between."
                   " Text for all columns will be concatenated as one.")
@click.option('--y_train_col', default='y_train', show_default=True,
              help="y_train column name. There must be 1 label per instance.")
@click.option('--stop_lang', default='en', show_default=True,
              help="Stop words language. For full list go to: https://github.com/stopwords-iso.")
@click.option('--n_features', default='10', show_default=True,
              help="Get the top n_features only.")
@click.option('--dir_save', default='fittedMods', show_default=True,
              help="create folder 'trained_models/[f_name]' to store fitted model.")

def main(file_path, x_train_col, y_train_col, stop_lang, n_features, dir_save):
    '''
    Application to calculate the support and top coefficients (log of the odds) for
    each class in dataset. Results are displayed and saved as CSV files.
    Fitted models are also saved.
    '''

    # Pipeline to process corpus
    pipeline = Pipeline([
        ('tfIdf', TfidfVectorizer(stop_words=getStopWords(stop_lang), analyzer='word')),
        ('logModel', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=100, n_jobs=-1))
    ])

    # read in corpus
    if '.json' in file_path:
        corpus = pd.read_json(file_path, lines=True)
    else:
        corpus = pd.read_csv(file_path)

    # prepare X and y train sets
    X_train, y_train = prepareXyTrain(corpus, x_train_col, y_train_col)

    # produce a fitting function to timeit
    @decorator_timeit
    def fitter(pipeline, X, y):
        return pipeline.fit(X, y)

    print('fitting models...')
    fittedPipe = fitter(pipeline, X_train, y_train)

    print('preparing support and coefficients...')
    supportDf = getSupport(y_train)
    coeffsDf = getCoeffMatrix(fittedPipe, int(n_features))

    print("\n=== TOP FEATURES ===")
    print(coeffsDf.head(int(n_features)))
    print("\n=== SUPPORT ===")
    print(supportDf.head(supportDf.shape[0]))
    print()

    # save trained model and dataframes
    modelSaver(dir_save, fittedPipe, supportDf, coeffsDf)


if __name__ == '__main__':
    main()
