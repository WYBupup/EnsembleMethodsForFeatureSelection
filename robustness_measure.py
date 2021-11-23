from abc import ABCMeta, abstractmethod
import scipy.stats
import numpy as np
"""
RobustnessMeasure().run_and_set_in_result(feature selection, result, result_index)

doing robustness analysis for feature selection(ranks)(D,cv)'s every combinations of different folds(cv(cv-1)/2) and 
set all the results in result[result index]

For different Robustness measure the similarity of two ranks is different
Dummy Spearman Jaccard_Index
"""


class Measure(metaclass=ABCMeta):
    def __init__(self):
        self.__name__ = type(self).__name__

    def run_and_set_in_results(self, features_selection, results, result_index):
        """
        feature selection (D,cv) are the ranks
        result[result_index]= robustness measurement of every combination of feature selections (k(k-1)/2,)
        """
        #  features_selection ranks(D,K )D:features dimension K:times of feature selection method for one data set
        np.random.seed()
        results[result_index] = self.measures(features_selection)

    @abstractmethod
    # features ranks is matrix with each rows represent a feature, and the columns its rankings  (D,K)
    #  bigger ranking means the feature is more important
    def measures(self, features_ranks):
        pass


class RobustnessMeasure(Measure, metaclass=ABCMeta):
    def measures(self, features_ranks):
        """
        features_ranks (D,cv)
        return robustness which is the robustnesss of all combinations of feature ranks (k(k-1)/2,)
        """
        #  robustness store robust measure of every combination of results(K) of feature selection method
        robustness = []
        for i in range(1, features_ranks.shape[1]):
            for j in range(i):
                robustness.append(self.robustness(features_ranks[:, i], features_ranks[:, j]))
        return robustness

    @abstractmethod
    def robustness(self, ranks1, ranks2):
        pass


class Dummy(RobustnessMeasure):
    def robustness(self, ranks1, ranks2):
        return 1


class Spearman(RobustnessMeasure):
    def __init__(self):
        super().__init__()
        self.__name__ = "Spearman Coefficient"

    def robustness(self, ranks1, ranks2):
        return scipy.stats.spearmanr(ranks1, ranks2)[0]


class JaccardIndex(Measure):
    def __init__(self, percentage=0.1):
        super().__init__()
        self.percentage = percentage
        self.__name__ = "Jaccard Index {:.2%}".format(percentage)

    def measures(self, features_ranks):
        if np.any(np.min(features_ranks, axis=0) != np.ones(features_ranks.shape[1], dtype=np.int32)):
            print(features_ranks)
            raise ValueError('features_rank ranking does not always begin with a 1')

        # the minimal rank a feature mast have to be chose
        minimal_rank = int((1 - self.percentage) * features_ranks.shape[0]) + 1

        # set everything below the minimal rank to zero and everything else to 1
        features_ranks[features_ranks < minimal_rank] = 0
        features_ranks[0 != features_ranks] = 1

        k = features_ranks.shape[1]
        #  jaccard_indices store jaccard index of every combination of results(K) of feature selection method
        jaccard_indices = []

        # jaccard_indices is symmetric
        for i in range(1, k):
            for j in range(i):
                f_i = features_ranks[:, i] == 1
                f_j = features_ranks[:, j] == 1

                intersection = np.logical_and(f_i, f_j)
                union = np.logical_or(f_i, f_j)

                jaccard_indices.append(np.sum(intersection) / np.sum(union))

        return jaccard_indices



