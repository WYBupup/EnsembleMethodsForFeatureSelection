from data_sets import DataSets
from robustness_measure import Measure
from abc import ABCMeta, abstractmethod
import numpy as np

"""
GoodnessMeasure(data_set_name, n_indices).run_and_set_in_result(feature_selection, result, result_index)

get the labeled features(if features are not labeled, return k's 0) of data set, and base on the the feature ranks(D,cv)
and n_significant_features and n_indices to compute the precision of feature selection for every fold (cv) and set the result 
in result[result_index]
"""


class RankData:
    def __init__(self, features_rank, n_significant_features, n_indices):
        """
        :param features_rank:  the ranks of the features (D,) and the features are already ordered so that
        features[:n_significant] is the relevant feature while features[n_significant_features:] is irrelevant
        features
        :param n_significant_features: the number of relevant features
        :param n_indices: the number of features selected
        """

        #  bigger rank means the feature is more important
        self.features_rank = features_rank
        self.sorted_indices = np.argsort(features_rank)[::-1]  # rank the feature rank in descending order
        self.n_significant = n_significant_features
        self.n_indices = n_indices

    def __len__(self):
        return len(self.features_rank)

    @property
    def true_positive(self):
        return (self.sorted_indices[:self.n_indices] < self.n_significant).sum()

    @property
    def false_positive(self):
        return (self.sorted_indices[:self.n_indices] >= self.n_significant).sum()

    @property
    def true_negative(self):
        return (self.sorted_indices[self.n_indices:] >= self.n_significant).sum()

    @property
    def false_negative(self):
        return (self.sorted_indices[self.n_indices:] < self.n_significant).sum()


class GoodnessMeasure(Measure, metaclass=ABCMeta):
    """
    the goodness measurement of a data set with the feature rankings  (k)
    parent function run_ans_set(ranks, result, result_index):
    result[result_index]= accuracy of the ranks (k,)
    """

    def __init__(self, data_set_name, n_indices=None):
        """
        get number of relevant features for one data setand number of features selected
        """
        super().__init__()
        feature_probe_labels = DataSets.load_features_labels(data_set_name)
        if feature_probe_labels is None:
            self.n_significant_features = None
        else:
            self.n_significant_features = np.sum([feature_probe_labels == 1])
        self.n_indices = self.n_significant_features if n_indices is None else n_indices
        self.__name__ = self.__name__ + str(n_indices)

    def measures(self, features_ranks):
        """
        features_ranks (D,cv)
        :return: the accuracy of feature ranks (labeled) (k,)
        """
        if not self.n_significant_features:
            return 0

        goodness = []
        for i in range(features_ranks.shape[1]):
            goodness.append(self.goodness(
                RankData(features_ranks[:, i].T, self.n_significant_features, self.n_indices)
            ))
        return np.array(goodness)

    @abstractmethod
    def goodness(self, data: RankData):
        pass


class Dummy(GoodnessMeasure):
    def __init__(self, *args, n_significant_features=None, **kwargs):
        super().__init__(*args, **kwargs)
        if n_significant_features is not None:
            self.n_significant_features = n_significant_features

    def goodness(self, data: RankData):
        return data.features_rank[0]


class Accuracy(GoodnessMeasure):
    def goodness(self, data: RankData):
        return (data.true_negative + data.true_positive) / len(data)


class Precision(GoodnessMeasure):
    def goodness(self, data: RankData):
        return data.true_positive / data.n_indices


class XPrecision(GoodnessMeasure):
    def goodness(self, data: RankData):
        p = 0
        alpha = 0.5
        n = data.n_significant
        for i in range(data.sorted_indices.shape[0] // n):
            positives = (data.sorted_indices[n * i: n * (i + 1)] < n).sum() / n
            p += alpha ** i * positives
        return p


class RankingLoss(GoodnessMeasure):
    def goodness(self, data: RankData):
        indices = []
        for i in range(data.n_significant):
            indices.append(int(np.argwhere(data.sorted_indices == i)))
        rs = (np.max(indices) - (data.n_significant - 1)) / (len(data) - data.n_significant)
        return rs


class LastRelevantFeature(GoodnessMeasure):
    def goodness(self, data: RankData):
        indices = []
        for i in range(data.n_significant):
            indices.append(int(np.argwhere(data.sorted_indices == i)))
        return np.max(indices)