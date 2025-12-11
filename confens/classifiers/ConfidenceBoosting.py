import copy

import numpy

from confens.classifiers.ConfidenceEnsemble import ConfidenceEnsemble
from confens.utils.classifier_utils import get_classifier_name, predict_confidence


class ConfidenceBoosting(ConfidenceEnsemble):
    """
    Class for creating Confidence Boosting ensembles
    """

    def __init__(self, clf, n_base: int = 20, learning_rate: float = 3.0,
                 sampling_ratio: float = 0.7, relative_boost_thr: float = 0.8, static_boost_thr: float = None,
                 conf_thr: float = None, perc_decisors: float = 1.0,
                 n_decisors: int = None, weighted: bool = True):
        """
        Constructor
        :param clf: the algorithm to be used for creating base learners
        :param n_base: number of base learners (= size of the ensemble)
        :param learning_rate: learning rate for updating dataset weights
        :param sampling_ratio: percentage of the dataset to be used at each iteration
        :param boost_thr: threshold of acceptance for confidence scores. It is the percentile of confidence scores of the base estimator that are considered "confident enough"
        :param static_boost_thr: static threshold of acceptance for confidence scores. Lower confidence means untrustable result
        :param conf_thr: float value for confidence threshold
        :param perc_decisors: percentage of base learners to be used for prediction
        :param n_decisors: number of base learners to be used for prediction
        :param weighted: True if prediction has to be computed as a weighted sum of probabilities
        """
        super().__init__(clf, n_base, conf_thr, perc_decisors, n_decisors, weighted)

        # Boosting thresholds
        self.relative_boost_thr = relative_boost_thr
        self.static_boost_thr = static_boost_thr

        # Other ConfBoost parameters
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 2
        if sampling_ratio is not None:
            self.sampling_ratio = sampling_ratio
        else:
            self.sampling_ratio = 1 / n_base ** (1 / 2)

    def fit_ensemble(self, X, y=None):
        """
        Training function for the confidence boosting ensemble
        :param y: labels of the train set (optional, not required for unsupervised learning)
        :param X: train set
        """
        train_n = len(X)
        samples_n = int(train_n * self.sampling_ratio)
        weights = numpy.full(train_n, 1 / train_n)

        for learner_index in range(0, self.n_base):

            # Draw samples
            sample_x, sample_y = self.draw_samples(X, y, samples_n, weights)

            # Train learner
            learner = copy.deepcopy(self.clf_list[learner_index % len(self.clf_list)])
            learner.fit(sample_x, sample_y)
            if hasattr(learner, "X_"):
                learner.X_ = None
            if hasattr(learner, "y_"):
                learner.y_ = None
            self.estimators_.append(learner)

            # Derive boost threshold
            y_conf = predict_confidence(learner, X)
            if self.static_boost_thr is not None and 0 < self.static_boost_thr < 1:
                p_thr = self.static_boost_thr
            else:
                p_thr = numpy.sort(y_conf)[int((1-self.relative_boost_thr)*len(y_conf))] if y_conf is not None else 0.8

            # Update Weights
            update_flag = numpy.where(y_conf >= p_thr, 0, 1)
            weights = weights * (1 + self.learning_rate * update_flag)
            weights = weights / sum(weights)

    def classifier_name(self):
        """
        Gets classifier name as string
        :return: the classifier name
        """
        clf_name = get_classifier_name(self.clf)
        if self.weighted:
            return "ConfidenceBoosterWeighted(" + str(clf_name) + "-" + \
                   str(self.n_base) + "-" + str(self.relative_boost_thr) + "-" + str(self.static_boost_thr) + "-" + \
                   str(self.learning_rate) + "-" + str(self.sampling_ratio) + "-" + \
                   str(self.conf_thr) + "-" + str(self.perc_decisors) + "-" + str(self.n_decisors) + ")"
        else:
            return "ConfidenceBooster(" + str(clf_name) + "-" + \
                   str(self.n_base) + "-" + str(self.relative_boost_thr) + "-" + str(self.static_boost_thr) + "-" + \
                   str(self.learning_rate) + "-" + str(self.sampling_ratio) + "-" + \
                   str(self.conf_thr) + "-" + str(self.perc_decisors) + "-" + str(self.n_decisors) + ")"
