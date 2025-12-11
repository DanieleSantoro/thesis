import copy

import numpy
from pyod.models.base import BaseDetector
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array
from xgboost import XGBClassifier

# ---------------------------------- SUPPORT METHODS ------------------------------------
from confens.metrics.EnsembleMetric import get_default


def get_feature_importance(clf):
    if isinstance(clf, Classifier):
        return clf.feature_importances()
    else:
        return clf.feature_importances_


def auto_bag_rate(n_features):
    """
    Method used to automatically devise a rate to include features in bagging
    :param n_features: number of features in the dataset
    :return: the rate of features to bag
    """
    if n_features < 20:
        bag_rate = 0.8
    elif n_features < 50:
        bag_rate = 0.7
    elif n_features < 100:
        bag_rate = 0.6
    else:
        bag_rate = 0.5
    return bag_rate


def predict_uns_proba(uns_clf, test_features):
    proba = uns_clf.predict_proba(test_features)
    pred = numpy.argmax(proba, axis=1)
    for i in range(len(pred)):
        min_p = min(proba[i])
        max_p = max(proba[i])
        proba[i][pred[i]] = max_p
        proba[i][1 - pred[i]] = min_p
    return proba


class Classifier(ClassifierMixin, BaseEstimator):
    """
    Basic Abstract Class for Classifiers.
    Abstract methods are only the classifier_name, with many degrees of freedom in implementing them.
    Wraps implementations from different frameworks (if needed), sklearn and many deep learning utilities
    """

    def __init__(self, clf):
        """
        Constructor of a generic Classifier
        :param clf: algorithm to be used as Classifier
        """
        super().__init__()
        self.clf = copy.deepcopy(clf) if clf is not None else None
        self._estimator_type = "classifier"
        self.feature_importances_ = None
        self.X_ = None
        self.y_ = None

    def fit(self, X, y=None):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit + other data
        if y is not None:
            self.classes_ = unique_labels(y)
        else:
            self.classes_ = [0, 1]
        # self.X_ = X
        # self.y_ = y

        # Train clf
        self.clf.fit(X, y)
        self.feature_importances_ = self.compute_feature_importances()

        # Return the classifier
        return self

    def predict(self, X):
        """
        Method to compute predict of a classifier
        :return: array of predicted class
        """
        probas = self.predict_proba(X)
        if self.is_unsupervised():
            return numpy.argmax(probas, axis=1)
        else:
            return self.classes_[numpy.argmax(probas, axis=1)]

    def predict_proba(self, X):
        """
        Method to compute probabilities of predicted classes
        :return: array of probabilities for each classes
        """

        # Check if fit has been called
        check_is_fitted(self)
        X = check_array(X)

        return self.clf.predict_proba(X)

    def decision_function(self, X):
        """
        Compatibility with PYOD
        :param X: the test set
        :return: a numpy array, or None
        """
        if X is None:
            return None
        X = check_array(X)
        probas = self.predict_proba(X)
        if probas.shape[1] >= 2:
            a = probas[:, 1] / probas[:, 0]
            return a
        else:
            return numpy.zeros(X.shape[0])

    def predict_confidence(self, X):
        """
        Method to compute confidence in the predicted class
        :return: max probability as default
        """
        probas = self.predict_proba(X)
        return numpy.max(probas, axis=1)

    def compute_feature_importances(self):
        """
        Outputs feature ranking in building a Classifier
        :return: ndarray containing feature ranks
        """
        if hasattr(self.clf, 'feature_importances_'):
            return self.clf.feature_importances_
        elif hasattr(self.clf, 'coef_'):
            return numpy.sum(numpy.absolute(self.clf.coef_), axis=0)
        return []

    def is_unsupervised(self):
        """
        true if the classifier is unsupervised
        :return: boolean
        """
        return hasattr(self, 'classes_') and numpy.array_equal(self.classes_, [0, 1])

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        return self.clf.__class__.__name__

    def get_diversity(self, X, y, metrics=None):
        """
        Returns diversity metrics. Works only with ensembles.
        :param metrics: name of the metrics to output (list of Metric objects)
        :param X: test set
        :param y: labels of the test set
        :return: diversity metrics
        """
        X = check_array(X)
        predictions = []
        check_is_fitted(self)
        if hasattr(self, "estimators_"):
            # If it is an ensemble and if it is trained
            for baselearner in self.estimators_:
                predictions.append(baselearner.predict(X))
            predictions = numpy.column_stack(predictions)
        elif self.clf is not None:
            # If it wraps an ensemble
            check_is_fitted(self.clf)
            if hasattr(self.clf, "estimators_"):
                # If it is an ensemble and if it is trained
                for baselearner in self.clf.estimators_:
                    predictions.append(baselearner.predict(X))
                predictions = numpy.column_stack(predictions)
        if predictions is not None and len(predictions) > 0:
            # Compute metrics
            metric_scores = {}
            if metrics is None or not isinstance(metrics, list):
                metrics = get_default()
            for metric in metrics:
                metric_scores[metric.get_name()] = metric.compute_diversity(predictions, y)
            return metric_scores
        else:
            # If it is not an ensemble
            return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self.clf, parameter, value)
        return self


class UnsupervisedClassifier(Classifier, BaseDetector):
    """
    Wrapper for unsupervised classifiers belonging to the library PYOD
    """

    def __init__(self, clf):
        """
        Constructor of a generic UnsupervisedClassifier. Assumes that clf is an algorithm from pyod
        :param clf: pyod algorithm to be used as Classifier
        """
        self.clf = clf
        self.contamination = clf.contamination
        self._estimator_type = "classifier"
        self.feature_importances_ = None
        self.X_ = None
        self.y_ = None

    def fit(self, X, y=None):
        # Store the classes seen during fit + other data
        self.classes_ = [0, 1]
        self.X_ = X
        self.y_ = None

        # Train clf
        self.clf.fit(X)
        self.feature_importances_ = self.compute_feature_importances()

        # Return the classifier
        return self

    def decision_function(self, X):
        """
        pyod function to override. Calls the wrapped classifier.
        :param X: test set
        :return: decision function
        """
        return self.clf.decision_function(X)

    def predict_proba(self, X):
        """
        Method to compute probabilities of predicted classes.
        It has to e overridden since PYOD's implementation of predict_proba is wrong
        :return: array of probabilities for each classes
        """

        # Check if fit has been called
        check_is_fitted(self)
        X = check_array(X)

        pred_score = self.decision_function(X)
        probs = numpy.zeros((X.shape[0], 2))
        if isinstance(self.contamination, (float, int)):
            pred_thr = pred_score - self.clf.threshold_
        min_pt = min(pred_thr)
        max_pt = max(pred_thr)
        anomaly = pred_thr > 0
        cont = numpy.asarray([pred_thr[i] / max_pt if anomaly[i] else (pred_thr[i] / min_pt if min_pt != 0 else 0.2)
                              for i in range(0, len(pred_thr))])
        probs[:, 0] = 0.5 + cont / 2
        probs[:, 1] = 1 - probs[:, 0]
        probs[anomaly, 0], probs[anomaly, 1] = probs[anomaly, 1], probs[anomaly, 0]
        return probs

    def classifier_name(self):
        """
        Returns the name of the classifier (as string)
        """
        return self.clf.__class__.__name__


class XGB(Classifier):
    """
    Wrapper for the xgboost.XGBClassifier algorithm
    Can be used as reference to see what's needed to extend the Classifier class
    """

    def __init__(self, n_estimators=100):
        Classifier.__init__(self, XGBClassifier(n_estimators=n_estimators))
        self.l_encoder = None

    def fit(self, X, y=None):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Store the classes seen during fit + other data
        self.classes_ = unique_labels(y)
        self.l_encoder = LabelEncoder()
        y = self.l_encoder.fit_transform(y)

        # self.X_ = X
        # self.y_ = y

        # Train clf
        self.clf.fit(X, y)
        self.feature_importances_ = self.compute_feature_importances()

        # Return the classifier
        return self

    def classifier_name(self):
        return "XGBClassifier(" + str(self.clf.n_estimators) + ")"
