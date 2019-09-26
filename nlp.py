import click
from functools import partial

from logistic import logistic_main
from lda import lda_main
from functions.helpers import rawAnalyzer
from functions.helpers import getStopWords

# after spacy download:
# python -m spacy download en_core_web_sm
import en_core_web_sm


@click.command()
@click.option('--x_train', default='X_train', show_default=True,
              help="X_train column name. Multiple column names can be passed with '*' between."
                   " Text for all columns will be concatenated as one.")
@click.option('--y_train', default='y_train', show_default=True,
              help="y_train column name. There must be 1 label per instance.")
@click.option('--n_features', default='20', show_default=True,
              help="Get the top n_features.")
@click.option('--lda_components', default='',
              help="Range of LDA components to process. Example: '2-6'.")
def main(x_train, y_train, n_features, lda_components):
    """
    Topic modeling application. Modes:

    --> Supervised (default mode):

    Logistic Regression is used to get top coefficients.
    Application prints and saves the top coefficients (log of the odds), the classes support,
    and the trained model in the 'output' directory.

    Usage:
    place your file (csv of json) in the 'input' directory (create 1 in root if necessary) and
    specify your X and y labels using the command line options. Application expects 1 file only,
    if multiple files are present only the first one is considered. To run navigate to root directory
    and type "python nlp.py" + options, e.g.:

    "python nlp.py --x_train text --y_train target"

    --> Unsupervised:

    latent Dirichlet allocation (LDA) is used with multiple components
    (topics). It is activated when passing the argument 'lda_components'. Multiple files
    can be processed (csv or json). A directory is created inside the 'output' directory
    per file processed. Output includes, per file, per number of topics:

    \b
    - Count of documents with topic (png file).
    - Percentage of documents with topic (png file).
    - Total percentage across documents normalized (png file).
    - Top features/words (csv file).

    Also, per file:

    \b
    - Line plot with likelihood for all number of topics (png file).
    - Line plot with perplexity for all number of topics (png file).

    Usage:
    place your file or files in the 'input' directory and pass the range of topics
    desired using the 'lda_components' option, e.g.:

    "python nlp.py --x_train text --lda_components 2-8"
    """
    # model used for lemmatization
    print('loading Spacy model...')
    spacy_model = en_core_web_sm.load()
    print('building vectorizer analyzer...')
    analyzer = partial(rawAnalyzer, spacy_model, getStopWords(spacy_model))

    if lda_components:
        print('--> LDA path...')
        lda_main(analyzer, lda_components, x_train, int(n_features))
    else:
        print('--> Logistic Regression path...')
        logistic_main(analyzer, x_train, y_train, int(n_features))


if __name__ == '__main__':
    main()
