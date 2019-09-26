# NLP Topic Modeling
Topic modeling application to extract top features/topics. It supports supervised 
and unsupervised modes.

## Usage
- You can use the ***nlp_env.yml*** file to create your conda environment. Example:
```console
conda env create -f /home/user/Desktop/nlp_env.yml
```
Otherwise you can manually install the needed packages (Anaconda is recommended). 
- **IMPORTANT:** after installing Spacy you must download the required model running:
```console
python -m spacy download en_core_web_sm
```
- To see all options go to the root directory and run:
```console
python nlp.py --help
```


```console
Usage: nlp.py [OPTIONS]

  Topic modeling application. Modes:

  --> Supervised (default mode):

  Logistic Regression is used to get top coefficients. Application prints
  and saves the top coefficients (log of the odds), the classes support, and
  the trained model in the 'output' directory.

  Usage: place your file (csv of json) in the 'input' directory and specify
  your X and y labels using the command line options. Application expects 1
  file only, if multiple files are present only the first one is considered.
  To run navigate to root directory and type "python nlp.py" + options,
  e.g.:

  "python nlp.py --x_train text --y_train target"

  --> Unsupervised:

  latent Dirichlet allocation (LDA) is used with multiple components
  (topics). It is activated when passing the argument 'lda_components'.
  Multiple files can be processed. A directory is created inside the
  'output' directory per file processed. Output includes, per file, per
  number of topics:

  - Count of documents with topic (png file).
  - Percentage of documents with topic (png file).
  - Total percentage across documents normalized (png file).
  - Top features/words (csv file).

  Also, per file:

  - Line plot with likelihood for all number of topics (png file).
  - Line plot with perplexity for all number of topics (png file).

  Usage: place your file or files in the 'input' directory and pass the
  range of topics desired using the 'lda_components' option, e.g.:

  "python nlp.py --x_train text --lda_components 2-8"

Options:
  --x_train TEXT         X_train column name. Multiple column names can be
                         passed with '*' between. Text for all columns will be
                         concatenated as one.  [default: X_train]
  --y_train TEXT         y_train column name. There must be 1 label per
                         instance.  [default: y_train]
  --n_features TEXT      Get the top n_features.  [default: 20]
  --lda_components TEXT  Range of LDA components to process. Example: '2-6'.
  --help                 Show this message and exit.

```

## To Do
- [] Validate options before computing.
- [] Add output examples to documentation.
- [] Use cross validation to get LDA scores (perhaps).
- [] Add module for Logistic Regression prediction. 

