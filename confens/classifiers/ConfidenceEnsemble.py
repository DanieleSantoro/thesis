import copy
from collections.abc import Iterable
from multiprocessing.pool import ThreadPool

import numpy
from pyod.models.base import BaseDetector
from sklearn.base import is_classifier, BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from confens.metrics.EnsembleMetric import get_default
from confens.utils.classifier_utils import predict_proba, predict_confidence, get_classifier_name


def define_bin_proba_thr(probs, cont: float = None) -> float:
    """
    Method for finding a confidence threshold based on the expected contamination (iterative)
    :param probs: probabilities to find threshold of
    :param target: the quantity to be used as reference for gettng to the threshold
    :return: a float value to be used as threshold to decide on anomalies (binary classification)
    """
    an_probas = numpy.sort(probs[:, 1])
    cutoff_index = int(len(an_probas)*(1-cont))
    thr = an_probas[cutoff_index]
    return thr


def define_proba_thr(probs, target: float = None, delta: float = 0.01) -> float:
    """
    Method for finding a confidence threshold based on the expected contamination (iterative)
    :param probs: probabilities to find threshold of
    :param target: the quantity to be used as reference for gettng to the threshold
    :param delta: the tolerance to stop recursion
    :return: a float value to be used as threshold for updating weights in boosting
    """
    target_cont = target
    p_thr = 0.5
    left_bound = 0.5
    right_bound = 1
    actual_cont = numpy.average(probs[:, 0] < p_thr)
    while abs(actual_cont - target_cont) > delta and abs(right_bound - left_bound) > 0.01:
        if actual_cont < target_cont:
            left_bound = p_thr
            p_thr = (p_thr + right_bound) / 2
        else:
            right_bound = p_thr
            p_thr = (p_thr + left_bound) / 2
        actual_cont = numpy.average(probs[:, 0] < p_thr)
    return p_thr


class ConfidenceEnsemble(ClassifierMixin, BaseEstimator):
    """
    Class for creating confidence ensembles
    """

    def __init__(self, clf, n_base: int = 10, conf_thr: float = None, perc_decisors: float = None,
                 n_decisors: int = None, weighted: bool = False):
        """
        Constructor
        :param clf: the algorithm(s) to be used for creating base learners
        :param n_base: number of base learners (= size of the ensemble)
        :param conf_thr: float value for confidence threshold
        :param perc_decisors: percentage of base learners to be used for prediction
        :param n_decisors: number of base learners to be used for prediction
        :param weighted: True if prediction has to be computed as a weighted sum of probabilities
        """
        super().__init__()
        self.clf = clf
        self._estimator_type = "classifier"
        self.feature_importances_ = None
        self.X_ = None
        self.y_ = None
        self.clf_list = []
        self.input_val = False
        self.weighted = weighted
        self.bin_proba_thr = None
        self.conf_thr = conf_thr
        self.contamination = clf.contamination if hasattr(clf, 'contamination') else None
        self.perc_decisors = perc_decisors
        self.n_decisors = n_decisors
        self.n_base = n_base
        self.estimators_ = []

    def validate_input(self):
        # Setting up classifiers to create base-learners
        self.clf = copy.deepcopy(self.clf) if self.clf is not None else None
        if is_classifier(self.clf) or isinstance(self.clf, BaseDetector):
            self.clf_list = [self.clf]
        elif isinstance(self.clf, Iterable):
            self.clf_list = []
            for clf_item in self.clf:
                if is_classifier(clf_item) or isinstance(clf_item, BaseDetector):
                    self.clf_list.append(clf_item)
                else:
                    print("Cant recognize object s a classifier")
            if len(self.clf_list) == 0:
                self.clf_list = [RandomForestClassifier(n_estimators=10)]
                print("clf is not a classifier. Using a 10-tree Random Forest as Base estimator")
        else:
            self.clf_list = [RandomForestClassifier(n_estimators=10)]
            print("clf is not a classifier. Using a 10-tree Random Forest as Base estimator")

        # N base estimators
        if self.n_base is None or self.n_base <= 1:
            print("Ensembles have to be at least 2")
            self.n_base = 10

        # Decisors
        if self.perc_decisors is not None and 0 < self.perc_decisors <= 1:
            if self.n_decisors is not None and 0 < self.n_decisors <= self.n_base:
                print('Both perc_decisors and n_decisors are specified, prioritizing perc_decisors')
            self.n_decisors = int(self.n_base * self.perc_decisors) if int(self.n_base * self.perc_decisors) > 0 else 1
        elif self.n_decisors is not None and 0 < self.n_decisors <= self.n_base:
            self.n_decisors = self.n_decisors
        elif self.conf_thr is None:
            self.n_decisors = self.n_base
        else:
            self.n_decisors = None

        # END
        self.input_val = True

    def fit(self, X, y=None):
        """
        Training function for the classifier
        :param y: labels of the train set (optional, not required for unsupervised learning)
        :param X: train set
        """
        if y is not None:
            X, y = check_X_y(X, y)
        else:
            X = check_array(X)
        self.classes_ = unique_labels(y) if y is not None else numpy.array([0, 1])
        if not self.input_val:
            self.validate_input()
        self.fit_ensemble(X, y)
        self.bin_proba_thr = define_bin_proba_thr(predict_proba(self, X), self.contamination) \
            if self.contamination is not None else 0.5

        # Compliance with SKLEARN and PYOD
        self.X_ = X[[0, 1], :]
        self.y_ = y
        self.decision_scores_ = self.decision_function(X)
        self.feature_importances_ = self.compute_feature_importances()

    def fit_ensemble(self, X, y=None):
        """
        Training function for the confidence ensemble: TO BE OVERRIDDEN
        :param y: labels of the train set (optional, not required for unsupervised learning)
        :param X: train set
        """
        pass

    def predict_base(self, X, learner_index):
        if hasattr(self, "feature_sets"):
            # ConfBag, each estimator uses a subset of features
            predictions = predict_proba(self.estimators_[learner_index], X[:, self.feature_sets[learner_index]])
        else:
            # ConfBoost, all estimators use all features
            predictions = predict_proba(self.estimators_[learner_index], X)
        return predictions

    def predict_proba(self, X, get_base:bool = False):
        """
        Function to assign probabilities to each class given a test set
        :param get_base: True if probabilities of base-learners have to be returned as well
        :param X: the test set
        :return: a numpy matrix
        """
        # Scoring probabilities and confidence
        proba_array = []
        conf_array = []
        base_dict = {}
        with ThreadPool(self.n_base) as p:
            proba_array = p.starmap(self.predict_base, [(X, i) for i in range(0, self.n_base)])
        for i in range(0, self.n_base):
            predictions = proba_array[i]
            base_dict[str(i) + '#' + get_classifier_name(self.estimators_[i])] = predictions
            conf_array.append(numpy.max(predictions, axis=1))
        # 3d matrix (clf, row, probability for class)
        proba_array = numpy.asarray(proba_array)
        # 2dim matrix (clf, confidence for row)
        conf_array = numpy.asarray(conf_array)

        # Compute final probabilities
        # Adjust probabilities with confidence if weighted
        if self.weighted:
            for i in range(0, X.shape[0]):
                proba_array[:, i, :] = (proba_array[:, i, :].T * conf_array[:, i]).T

        if self.n_decisors is not None:
            # Option 1: either n_decisors or perc_decisors is set, or conf_thr is None
            if self.n_decisors > self.n_base or self.n_decisors <= 0:
                # Checks for strange n_decisors values
                self.n_decisors = self.n_base
            if self.n_decisors == self.n_base:
                # Easy case, all contribute, no need to additional computation
                proba = numpy.average(proba_array, axis=0)
                proba = proba / numpy.sum(proba, axis=1).reshape(-1, 1)
            else:
                # Here it requires sorting confidences for understanding the "best" base-learners
                conf_array = conf_array.transpose()
                all_conf = -numpy.sort(-conf_array, axis=1)
                conf_thrs = all_conf[:, self.n_decisors - 1]
                mask = numpy.asarray(conf_array >= conf_thrs.reshape(-1, 1)).T
                masked_probas = numpy.asarray([numpy.multiply(proba_array[:, :, i], mask) for i in range(0, len(self.classes_))])
                p_sum = numpy.sum(masked_probas, axis=1)
                proba = (p_sum/numpy.sum(p_sum, axis=0)).T
        else:
            # Option 2: conf_thr is not None, neither n_decisors nor perc_decisors are set
            # Thus, base-learners contribute if they are confident at least 'conf_thr'
            conf_array = conf_array.transpose()
            mask = numpy.asarray(conf_array >= self.conf_thr).T
            masked_probas = numpy.asarray(
                [numpy.multiply(proba_array[:, :, i], mask) for i in range(0, len(self.classes_))])
            p_sum = numpy.sum(masked_probas, axis=1)
            proba = (p_sum / numpy.sum(p_sum, axis=0)).T

        # Final averaged Result
        if get_base:
            return proba, base_dict
        else:
            return proba

    def draw_samples(self, X, y, samples_n: int, weights = None):
        """
        Returns samples of a labeled set (X, y), y may be None
        :param weights: may be none, weights for sampling data (needed mostly for ConfBoost)
        :param X: the set to sample
        :param y: the labels to sample
        :param samples_n: numer of samples
        :return: a subset of (X, y)
        """
        if weights is None:
            indexes = numpy.random.choice(X.shape[0], samples_n, replace=False, p=None)
        else:
            indexes = numpy.random.choice(len(weights), samples_n, replace=False, p=weights)
        sample_x = numpy.asarray(X[indexes, :])
        # If data is labeled we also have to refine labels
        if y is not None and hasattr(self, 'classes_') and self.classes_ is not None and len(self.classes_) > 1:
            sample_y = y[indexes]
            sample_labels = unique_labels(sample_y)
            missing_labels = [item for item in self.classes_ if item not in sample_labels]
            # And make sure that there is at least a sample for each class of the problem
            if missing_labels is not None and len(missing_labels) > 0:
                # For each missing class
                for missing_class in missing_labels:
                    miss_class_indexes = numpy.asarray(numpy.where(y == missing_class)[0])
                    new_sampled_index = numpy.random.choice(miss_class_indexes, None, replace=False, p=None)
                    X_missing_class = X[new_sampled_index, :]
                    sample_x = numpy.append(sample_x, [X_missing_class], axis=0)
                    sample_y = numpy.append(sample_y, missing_class)
        else:
            sample_y = None
        return sample_x, sample_y

    def predict_confidence(self, X):
        """
        Method to compute the confidence in predictions of a classifier
        :param X: the test set
        :return: array of confidence scores
        """
        conf = numpy.zeros(X.shape[0])
        for clf in self.estimators_:
            conf += predict_confidence(clf, X)
        return conf / self.n_base

    def get_feature_importances(self):
        """
        Placeholder, to be implemented if possible
        :return: feature importances (to be tested)
        """
        fi = []
        for clf in self.estimators_:
            c_fi = clf.get_feature_importances()
            fi.append(c_fi)
        fi = numpy.asarray(fi)
        return numpy.average(fi, axis=1)

    def predict(self, X, get_base: bool = False):
        """
        Method to compute predict of a classifier
        :param get_base: True if predictions of base-learners have to be provided as well
        :param X: the test set
        :return: array of predicted class and (if get_base = True) dictionary with base-learners predictions
        """
        if get_base is True:
            proba, base_probas = predict_proba(self, X, get_base=True)
            pred_base = [self.classes_[numpy.argmax(base_p, axis=1)] for base_p in base_probas.values()]
            return self.classes_[numpy.argmax(proba, axis=1)], dict(zip(base_probas.keys(), pred_base))
        else:
            proba = predict_proba(self, X)
            return self.classes_[numpy.argmax(proba, axis=1)]

    def decision_function(self, X):
        """
        Compatibility with PYOD
        :param X: the test set
        :return: a numpy array, or None
        """
        if X is None:
            return None
        X = check_array(X)
        probas = predict_proba(self, X)
        if probas.shape[1] >= 2:
            return numpy.sum(probas[:, 1:], axis=1)
        else:
            return numpy.zeros(X.shape[0])

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
        ens_pred, predictions = self.predict(X, True)
        predictions = numpy.column_stack(list(predictions.values()))
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
