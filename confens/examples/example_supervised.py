# Support libs
import random

import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.model_selection as ms
# Used to save a classifier and measure its size in KB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Name of the folder in which look for tabular (CSV) datasets
from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting

# Scikit-Learn algorithms
# The PYOD library contains implementations of unsupervised classifiers.
# Works only with anomaly detection (no multi-class)
# ------- GLOBAL VARS -----------

CSV_FILE = "sample_data/sample_data_arancino.csv"
# Name of the column that contains the label in the tabular (CSV) dataset
LABEL_NAME = 'multilabel'
# Percantage of test data wrt train data
TT_SPLIT = 0.5
# True if debug information needs to be shown
VERBOSE = True

# Set random seed for reproducibility
random.seed(42)
numpy.random.seed(42)


# ----------------------- MAIN ROUTINE ---------------------
# This files shows an example of confidence ensembles for supervised learning.
# Can be used as is, just change the details to load the dataset

if __name__ == '__main__':

    # if file is a CSV, it is assumed to be a dataset to be processed
    df = pandas.read_csv(CSV_FILE, sep=",")
    if VERBOSE:
        print("\n------------ DATASET INFO -----------------")
        print("Data Points in Dataset '%s': %d" % (CSV_FILE, len(df.index)))
        print("Features in Dataset: " + str(len(df.columns)))

    # Set up train test split excluding categorical values that some algorithms cannot handle
    # 1-Hot-Encoding or other approaches may be used instead of removing
    x_train, x_test, y_train, y_test = ms.train_test_split(df.drop(columns=[LABEL_NAME]).to_numpy(),
                                                           df[LABEL_NAME].to_numpy(),
                                                           test_size=TT_SPLIT, shuffle=True)
    # Creating classifiers
    classifier = LinearDiscriminantAnalysis()
    cb_clf = ConfidenceBoosting(clf=classifier, n_base=10, learning_rate=2,
                                sampling_ratio=0.5, conf_thr=0.8)

    # Exercising Classifier
    classifier.fit(x_train, y_train)
    clf_pred = classifier.predict(x_test)

    # Exercising ConfBoost
    cb_clf.fit(x_train, y_train)
    cb_pred = cb_clf.predict(x_test)

    print('LDA has accuracy of %.3f, whereas the ConfBoost(LDA) has accuracy of %.3f' %
          (metrics.accuracy_score(y_test, clf_pred), metrics.accuracy_score(y_test, cb_pred)))
