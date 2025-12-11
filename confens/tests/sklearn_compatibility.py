# Support libs
import random

import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.model_selection as ms
# Used to save a classifier and measure its size in KB
from sklearn.calibration import CalibratedClassifierCV
# Name of the folder in which look for tabular (CSV) datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TunedThresholdClassifierCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from confens.classifiers.ConfidenceBagging import ConfidenceBagging
from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting
# Scikit-Learn algorithms
# The PYOD library contains implementations of unsupervised classifiers.
# Works only with anomaly detection (no multi-class)
# ------- GLOBAL VARS -----------
from confens.utils.classifier_utils import get_classifier_name
from confens.utils.general_utils import current_ms

CSV_FILE = "sample_data/sample_data_arancino.csv"
# Name of the column that contains the label in the tabular (CSV) dataset
LABEL_NAME = 'multilabel'
# Percantage of test data wrt train data
TT_SPLIT = 0.5
# True if debug information needs to be shown
VERBOSE = True

FORCE_BINARY = True

# Set random seed for reproducibility
random.seed(42)
numpy.random.seed(42)

# ----------------------- MAIN ROUTINE ---------------------
# This files shows an example of confidence ensembles for supervised learning.
# Can be used as is, just change the details to load the dataset

def get_base():
    return RandomForestClassifier(n_estimators=10)

if __name__ == '__main__':

    # if file is a CSV, it is assumed to be a dataset to be processed
    df = pandas.read_csv(CSV_FILE, sep=",", nrows=10000)
    if VERBOSE:
        print("\n------------ DATASET INFO -----------------")
        print("Data Points in Dataset '%s': %d" % (CSV_FILE, len(df.index)))
        print("Features in Dataset: " + str(len(df.columns)))

    # Set up train test split excluding categorical values that some algorithms cannot handle
    # 1-Hot-Encoding or other approaches may be used instead of removing
    y = df[LABEL_NAME].to_numpy()
    if FORCE_BINARY:
        y = numpy.where(df[LABEL_NAME] == "normal", 0, 1)
    x_train, x_test, y_train, y_test = ms.train_test_split(df.drop(columns=[LABEL_NAME]).to_numpy(), y,
                                                           test_size=TT_SPLIT, shuffle=True)
    # Creating classifiers
    cboost_parameters = {'n_base': [5, 10, 20], 'perc_decisors': [0.3, 0.5, 0.8, 1.0]}
    classifiers = [
        get_base(),
        TunedThresholdClassifierCV(estimator=get_base()),
        ConfidenceBagging(clf=get_base()),
        ConfidenceBoosting(clf=get_base()),
        TunedThresholdClassifierCV(estimator=ConfidenceBoosting(clf=get_base())),
        CalibratedClassifierCV(estimator=ConfidenceBoosting(clf=get_base()), method='sigmoid'),
        CalibratedClassifierCV(estimator=ConfidenceBoosting(clf=get_base()), method='isotonic'),
        GridSearchCV(estimator=ConfidenceBoosting(clf=get_base()), param_grid=cboost_parameters)
    ]

    for classifier in classifiers:
        # Exercising Classifier
        clf_name = get_classifier_name(classifier)
        start_time = current_ms()
        classifier.fit(x_train, y_train)
        end_time = current_ms()
        clf_pred = classifier.predict(x_test)

        print('%s has accuracy of %.3f, training in %d ms' %
              (clf_name, metrics.accuracy_score(y_test, clf_pred), end_time-start_time))  # , metrics.accuracy_score(y_test, cb_pred)))
