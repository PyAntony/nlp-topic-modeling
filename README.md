# Logistic Regression Top Coefficients

Simple Python application to extract the top coefficients (words) for every target variable from a text dataset. Scikit-learn TfidfVectorizer and LogisticRegression models with default parameters are used. This application can be used to quickly peek over the highest ranked words within each class in your text dataset. Coefficients, classes support, and trained models are saved.

Highest ranked words are understood as the words (or *features* since text is tokenized by words) with the highest ***log(odds ratio)*** (also known as ***coefficients***). They can be understood as the odds by which the text analyzed (news, article, document, message, etc.) belongs to the positive class given that the feature is present.

## Usage

To see options navigate to the root directory and type:  
 - **python ml-train.py --help**

```console
Options:
  --file_path TEXT    Path of csv or json corpus file.  [default: data.csv]
  --X_train_col TEXT  X_train column name. Multiple column names can be passed
                      with '*' between. Text for all columns will be
                      concatenated as one.  [default: X_train]
  --y_train_col TEXT  y_train column name. There must be 1 label per instance.
                      [default: y_train]
  --stop_lang TEXT    Stop words language. For full list go to:
                      https://github.com/stopwords-iso.  [default: en]
  --n_features TEXT   Get the top n_features only.  [default: 10]
  --dir_save TEXT     create folder 'trained_models/[f_name]' to store fitted
                      model.  [default: fittedMods]
  --help              Show this message and exit.
```

Notice that there is a default name for the input file. You can rename your csv file as ***data.csv*** and put it in the root directory of this application. Otherwise you have to specify the full path of your data (json and csv extensions are accepted).

## Example

I am running an example with a dataset downloaded from: https://www.kaggle.com/crawford/20-newsgroups/download. It contains news headlines and the categories where they appeared. For this example I am concatenating 2 columns, *headline* and *short_description*:
- **python ml-train.py --file_path News.json --X_train_col headline*short_description --y_train_col category**

Part of the output example:
```console
=== TOP FEATURES ===
target          ARTS          ARTS & CULTURE          BLACK VOICES             BUSINESS           ...     WELLNESS              WOMEN          WORLD NEWS         WORLDPOST         
values       feature      odd        feature      odd      feature      odd     feature      odd  ...      feature      odd   feature      odd    feature     odd   feature      odd
0                art  16151.9         artist  21599.5        black   360972     krugman  1762.76  ...     fearless   8015.9     women  19575.3     killed  754.89      isis  1809.18
1             artist  7928.35            art  14988.8       rapper   1266.4    business  1626.31  ...      workout  1102.82  feminism   2407.2   rohingya  591.73   ukraine  1057.46
2            artists   665.62           book  5025.31      african  1170.25   marketing  1151.51  ...          gps   806.58  abortion  2183.51     attack  423.74    greece   583.55
3              opera   369.74        artists   822.87       racial    967.4         ceo    694.3  ...         yoga   665.74  feminist  1622.16      saudi  302.66      gaza    475.4
4            theatre   221.82          books   277.95       racist   470.15          24   237.57  ...       health   654.12    sexist   581.68      trump  276.36      iran    322.2
5            theater    190.2          trump   204.43       racism   196.73        wall   228.92  ...        study   526.63    female   382.22    myanmar  243.29     china   205.73
6            nighter   127.61       hamilton   145.15     ferguson   146.54     walmart   197.65  ...        sleep   478.01     woman   378.59     london  243.04     india   199.56
7        photography    104.8       broadway   143.86         race   130.99        uber   186.17  ...       cancer   424.45      rape   163.99      korea  241.87     egypt   190.77
8             ballet    71.25   photographer   120.82      atlanta   122.77   workplace   162.33  ...      fitness   310.15      body   153.43     police  230.55     yemen   169.27
9       photographer    68.36       feminist    83.56          lee   110.91  leadership    152.3  ...  researchers   296.66    sexism   151.66    country  185.07     ebola   159.16

[10 rows x 82 columns]

=== SUPPORT ===
           yTarget  count
0         POLITICS  32739
1         WELLNESS  17827
2    ENTERTAINMENT  16058
3           TRAVEL   9887
4   STYLE & BEAUTY   9649
5        PARENTING   8677
6   HEALTHY LIVING   6694
7     QUEER VOICES   6314
8     FOOD & DRINK   6226
9         BUSINESS   5937
10          COMEDY   5175
11          SPORTS   4884

```
