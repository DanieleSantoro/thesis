import copy
import random
from multiprocessing import Pool, Queue
from multiprocessing.pool import ThreadPool

from confens.classifiers.ConfidenceEnsemble import ConfidenceEnsemble
from confens.utils.classifier_utils import get_classifier_name
from confens.utils.general_utils import current_ms


class ConfidenceBagging(ConfidenceEnsemble):
    """
    Class for creating bagging ensembles
    """

    def __init__(self, clf, n_base: int = 10, max_features: float = 0.7, sampling_ratio: float = 0.7,
                 conf_thr: float = None, perc_decisors: float = None, n_decisors: int = None,
                 weighted: bool = False, parallel_train: bool = True):
        """
        Constructor
        :param clf: the algorithm to be used for creating base learners
        :param n_base: number of base learners (= size of the ensemble)
        :param max_features: percentage of features to be used at each iteration
        :param sampling_ratio: percentage of the dataset to be used at each iteration
        :param conf_thr: float value for confidence threshold
        :param perc_decisors: percentage of base learners to be used for prediction
        :param n_decisors: number of base learners to be used for prediction
        :param weighted: True if prediction has to be computed as a weighted sum of probabilities
        """
        super().__init__(clf, n_base, conf_thr, perc_decisors, n_decisors, weighted)
        self.max_features = max_features if max_features is not None and 0 < max_features <= 1 else 0.7
        self.sampling_ratio = sampling_ratio if sampling_ratio is not None and 0 < sampling_ratio <= 1 else 0.7
        self.feature_sets = []
        self.parallel_train = parallel_train

    def train_base_bagger(self, X, y, samples_n, features, learner_index):
        sample_x, sample_y = self.draw_samples(X, y, samples_n)
        sample_x = sample_x[:, features]
        if len(features) == 1:
            sample_x = sample_x.reshape(-1, 1)
        # Train learner
        learner = copy.deepcopy(self.clf_list[learner_index % len(self.clf_list)])
        learner.fit(sample_x, sample_y)
        if hasattr(learner, "X_"):
            learner.X_ = None
        if hasattr(learner, "y_"):
            learner.y_ = None
        # Test Learner
        return learner

    def fit_ensemble(self, X, y=None):
        train_n = len(X)
        bag_features_n = int(X.shape[1] * self.max_features)
        samples_n = int(train_n * self.sampling_ratio)
        # Drawing features
        for learner_index in range(0, self.n_base):
            features = random.sample(range(X.shape[1]), bag_features_n)
            features.sort()
            self.feature_sets.append(features)
        if self.parallel_train:
            # Training in parallel
            with ThreadPool(self.n_base) as p:
                self.estimators_ = p.starmap(self.train_base_bagger,
                                             [(X, y, samples_n, self.feature_sets[i], i) for i in range(0, self.n_base)])
        else:
            self.estimators_ = []
            for learner_index in range(0, self.n_base):
                self.estimators_.append(self.train_base_bagger(X, y, samples_n, self.feature_sets[learner_index], learner_index))

    def classifier_name(self):
        """
        Gets classifier name as string
        :return: the classifier name
        """
        clf_name = get_classifier_name(self.clf)
        if self.weighted:
            return "ConfidenceBaggerWeighted(" + str(clf_name) + "-" + str(self.n_base) + "-" + \
                   str(self.conf_thr) + "-" + str(self.perc_decisors) + "-" + str(self.n_decisors) + "-" + \
                   str(self.max_features) + "-" + str(self.sampling_ratio) + ")"
        else:
            return "ConfidenceBagger(" + str(clf_name) + "-" + str(self.n_base) + "-" + \
                   str(self.conf_thr) + "-" + str(self.perc_decisors) + "-" + str(self.n_decisors) + "-" + \
                   str(self.max_features) + "-" + str(self.sampling_ratio) + ")"
