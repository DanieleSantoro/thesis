# Support libs
import random

import numpy
import pandas
import sklearn.metrics as metrics
import sklearn.model_selection as ms
# Used to save a classifier and measure its size in KB
from pyod.models.copod import COPOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA
# Name of the folder in which look for tabular (CSV) datasets
from pyod.models.suod import SUOD

from confens.classifiers.ConfidenceBagging import ConfidenceBagging
from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting
# Scikit-Learn algorithms
# The PYOD library contains implementations of unsupervised classifiers.
# Works only with anomaly detection (no multi-class)
# ------- GLOBAL VARS -----------
from confens.utils.general_utils import get_classifier_name

CSV_FILE = "sample_data/sample_data_arancino.csv"
# Name of the column that contains the label in the tabular (CSV) dataset
LABEL_NAME = 'multilabel'
# Name of the 'normal' class in datasets. This will be used only for binary classification (anomaly detection)
NORMAL_TAG = 'normal'
# Percantage of test data wrt train data
TT_SPLIT = 0.5
# True if debug information needs to be shown
VERBOSE = True

# Set random seed for reproducibility
random.seed(42)
numpy.random.seed(42)

# ----------------------- MAIN ROUTINE ---------------------
# This files shows an example of confidence ensembles for unsupervised learning.
# Can be used as is, just change the details to load the dataset

if __name__ == '__main__':

    # if file is a CSV, it is assumed to be a dataset to be processed
    df = pandas.read_csv(CSV_FILE, sep=",", nrows=10000)
    if VERBOSE:
        print("\n------------ DATASET INFO -----------------")
        print("Data Points in Dataset '%s': %d" % (CSV_FILE, len(df.index)))
        print("Features in Dataset: " + str(len(df.columns)))

    # Set up train test split excluding categorical values that some algorithms cannot handle
    # 1-Hot-Encoding or other approaches may be used instead of removing
    y = numpy.where(df[LABEL_NAME] == "normal", 0, 1)
    an_perc = numpy.average(y)
    x_train, x_test, y_train, y_test = ms.train_test_split(df.drop(columns=[LABEL_NAME]).to_numpy(), y,
                                                           test_size=TT_SPLIT, shuffle=True)
    # Creating classifiers
    # Note that unsupervised classifiers in PYOD require a contamination
    classifiers = [PCA(contamination=an_perc),
                   ConfidenceBagging(clf=PCA(contamination=an_perc)),
                   ConfidenceBoosting(clf=PCA(contamination=an_perc)),
                   FeatureBagging(contamination=an_perc, base_estimator=ConfidenceBagging(clf=HBOS(contamination=an_perc)), n_estimators=5),
                   FeatureBagging(contamination=an_perc, base_estimator=ConfidenceBoosting(clf=HBOS(contamination=an_perc)), n_estimators=5),
                   SUOD(contamination=an_perc, base_estimators=[PCA(contamination=an_perc),
                                                                ConfidenceBagging(clf=COPOD(contamination=an_perc)),
                                                                ConfidenceBoosting(clf=COPOD(contamination=an_perc))])]
    for classifier in classifiers:
        #cb_clf = ConfidenceBoosting(clf=classifier, n_base=10, learning_rate=2,
        #                            sampling_ratio=0.5, conf_thr=0.8)

        # Exercising Classifier
        clf_name = get_classifier_name(classifier)
        classifier.fit(x_train)
        clf_pred = classifier.predict(x_test)

        # Exercising ConfBoost
        #cb_clf.fit(x_train)
        #cb_pred = cb_clf.predict(x_test)

        print('%s has accuracy of %.3f' %
              (clf_name, metrics.accuracy_score(y_test, clf_pred)))#, metrics.accuracy_score(y_test, cb_pred)))
