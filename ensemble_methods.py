from feature_selector import DataSetFeatureSelector
import numpy as np
from abc import ABCMeta, abstractmethod
from data_sets import DataSets, PreComputedData
from sklearn.model_selection import KFold
import multiprocessing
from accuracy_measure import ber
from io_utils import mkdir


class EnsembleMethod(DataSetFeatureSelector, metaclass=ABCMeta):
    max_parallelism = multiprocessing.cpu_count()

    def __init__(self, data_set_feature_selectors):
        """
        :param data_set_feature_selectors:  list of feature selectors defined in feature_selector.py
        initialize
            self.feature_selectors
            self.__name__= EnsembleMethod_[feature selectors]
        """
        super().__init__()

        if not isinstance(data_set_feature_selectors, list):
            data_set_feature_selectors = [data_set_feature_selectors]

        for data_set_feature_selector in data_set_feature_selectors:
            if not isinstance(data_set_feature_selector, DataSetFeatureSelector):
                raise ValueError("Only DataSetFeatureSelector can be used")

        self.feature_selectors = data_set_feature_selectors

        self.__name__ = "Heterogeneous_"+self.__name__+"_{}".format(self.fs_short_names())

    def fs_short_names(self):
        return "_".join(str(f) for f in self.feature_selectors)

    def rank_data_set(self, data_set, cv_generator):
        """
        :param data_set: name of data set
        :param cv_generator:  this is the function of class Benchmark.cv
        For the particular data set:
        for every feature selector in self.feature_selectors:
            compute the weight of features for different folds's training set (cv,D)
        get bench_features_selection (feature selectors, cv, D)
        for each cv fold combine the weight of features over different feature selectors (feature_selectors, D)->(D,) and
        rank the weight
        return result (cv,D)


        change save the ranks locally
        """
        super().rank_data_set(data_set, cv_generator)

        data, labels = DataSets.load(data_set)
        cv = cv_generator(labels.shape[0])  # generator generated by CV.split(np.arange(len(labels)))

        try:
            return PreComputedData.load(data_set, cv, "rank", self)
        except FileNotFoundError:
            print(f"=> Generating feature {'ranks'} of {data_set} ({cv.__name__}) with {self.__name__}")
            try:
                cv_indices = PreComputedData.load_cv(data_set, cv)
            except FileNotFoundError:
                mkdir(PreComputedData.cv_dir(data_set, cv))

                cv_indices = list(cv)
                np.save(PreComputedData.cv_file_name(data_set, cv), cv_indices)

            bench_features_selection = []
            for f in self.feature_selectors:
                bench_features_selection.append(f.weight_data_set(data_set, cv_generator))
            bench_features_selection = np.array(bench_features_selection)  # (feature_selectors, cv, D)

            feature_selection = multiprocessing.Manager().dict()

            with multiprocessing.Pool(processes=self.max_parallelism) as pool:
                for i in range(bench_features_selection.shape[1]):
                    pool.apply_async(
                        self.run_and_set_in_results,
                        kwds={
                            'results': feature_selection,
                            'result_index': i,
                            'feature_selection': bench_features_selection[:, i],
                            'data': data[:, cv_indices[i][0]],
                            'labels': labels[cv_indices[i][0]]
                        }
                    )
                pool.close()
                pool.join()

            ranks = [0 for _ in range(bench_features_selection.shape[1])]
            for i, rank in feature_selection.items():
                ranks[i] = rank

            ranks = np.array(ranks)
            self.__save(data_set, cv, "rank", ranks)

            return ranks

    def weight_data_set(self, data_set, cv_generator):
        # old version self.normalize(self.rank_data_set(data_set,cv_generator)) which will not appropriately weight the
        # weight since the return sze of rank_data_set is (cv, D) and the normalize will be implemented on cv axi
        super().weight_data_set(data_set, cv_generator)
        data, labels = DataSets.load(data_set)
        cv = cv_generator(labels.shape[0])
        try:
            return PreComputedData.load(data_set, cv, "weight", self)
        except FileNotFoundError:
            print(f"generate weights for {data_set}({cv.__name__}) with {self.__name__}")
            weights = self.normalize(self.rank_data_set(data_set, cv_generator).T).T
            self.__save(data_set, cv, "weight", weights)
        return weights

    def run_and_set_in_results(self, results, result_index, feature_selection, data, labels):
        """
        :param results:
        :param result_index:
        :param feature_selection: (feature selectors, D)
        :param data: the training data of one fold
        :param labels:  the training labels of one fold
        combine the feature weights of different feature selectors(feature_selectors, D)->(D,) and rank it
        set the result in result[result_index]
        """
        np.random.seed()
        results[result_index] = self.rank_weights(self.combine(feature_selection, data, labels))

    @abstractmethod
    def combine(self, feature_selection, data, labels):
        """
        :param feature_selection: (feature selectors, D)
        :param data: training set of fold
        :param labels: training labels of fold
        :return:
        """
        pass

    def __save(self, data_set, cv, method, feature_selection):
        """
        :param data_set: name of data set
        :param cv: cv generator
        :param method:  weights or rank(assessment method)
        :param feature_selection: the weights/ranks analyzed by feature selection for every fold's training set
        :return: save feature_selection to "../pre_computed_data/data_set/cv/assessment_method/feature_selector.npy"
        """
        mkdir(PreComputedData.dir_name(data_set, cv, method))
        np.save(PreComputedData.file_name(data_set, cv, method, self), feature_selection)


class Influence(EnsembleMethod):
    def __init__(self, k=1, bias=1, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.bias = bias
        if self.k != 1:
            self.__name__ += " k={}".format(self.k)
        if self.bias != 1:
            self.__name__ += " bias={}".format(self.bias)

    def combine(self, feature_selection, data, labels):
        """
        :param feature_selection: (feature selectors, D)
        :param data: training set of fold
        :param labels: training labels of fold
        :return:
        """
        return self._influence(feature_selection).mean(axis=0)

    def _influence(self, x):
        """
        :param x: (feature selectors, D)
        :return:
        """
        h = np.exp(np.arctanh(self.k * (x - self.bias)))
        return (h.T / h.sum(axis=1)).T


class Gibbs(EnsembleMethod):
    def __init__(self, k=0.1, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.__name__ += " k={}".format(self.k)

    def combine(self, feature_selection, data, labels):
        """
        :param feature_selection: (feature selectors, D)
        :param data: training set of fold
        :param labels: training labels of fold
        :return:
        """
        return self._influence(feature_selection).mean(axis=0)

    def _influence(self, x):
        h = np.exp(-x / self.k)
        return (h.T / h.sum(axis=1)).T


class InfluenceStd(EnsembleMethod):
    def combine(self, feature_selection, data, labels):
        feature_selection = np.exp(feature_selection)
        influence = (feature_selection.T / feature_selection.sum(axis=1)).T
        return influence.mean(axis=0) / (1 + influence.std(axis=0))


class Mean(EnsembleMethod):
    def combine(self, features_selection, data, labels):
        return features_selection.mean(axis=0)


class MeanNormalizedSum(EnsembleMethod):
    def combine(self, features_selection, data, labels):
        return self._norm(features_selection).mean(axis=0)

    def _norm(self, x):
        return np.nan_to_num((x.T / x.sum(axis=1)).T)


class MeanStd(EnsembleMethod):
    def __init__(self, power=1, **kwargs):
        super().__init__(**kwargs)
        self.__name__ = "Mean std - {}".format(power)
        self.power = power

    def combine(self, features_selection, data, labels):
        return np.power(features_selection, self.power).mean(axis=0) / features_selection.std(axis=0)


class SMean(EnsembleMethod):
    def __init__(self, min_mean_max=[1, 1, 1], **kwargs):
        super().__init__(**kwargs)
        self.weights = np.array(min_mean_max)
        self.__name__ = "SMean - {} {} {} ({})".format(*min_mean_max, self.fs_short_names())

    def combine(self, features_selection, data, labels):
        f_mean = np.mean(features_selection, axis=0)
        f_max = np.max(features_selection, axis=0)
        f_min = np.min(features_selection, axis=0)
        return (np.vstack((f_min, f_mean, f_max)) * self.weights[:, np.newaxis]).mean(axis=0)


class EmWithClassifier(EnsembleMethod, metaclass=ABCMeta):
    def __init__(self, classifiers, **kwargs):
        super().__init__(**kwargs)
        self.classifiers = classifiers

    def accuracy_fs(self, features_selection, data, labels):
        """
        :param features_selection: (feature selectors, D)
        :param data: training set of fold
        :param labels: labels of training set of fold
        For every feature selection :
            select the top 1% weighted features to measure the accuracy of (data,labels) using k-fold for every
            classifier(classifier, k-fold) and sum them to represent the  accuracy of this feature selector
        accuracy(Feature selectors)
        """
        cv = KFold().split(np.arange(labels.shape[0]))  # generator by 5-Fold
        accuracy = np.zeros(len(self.feature_selectors))

        for i in range(len(self.feature_selectors)):
            best_features_indices = np.argsort(features_selection[i])[:-int(features_selection[i].shape[0] / 100):-1]
            for train_index, test_index in cv:
                for c in self.classifiers:
                    c.fit(data[np.ix_(best_features_indices, train_index)].T, labels[train_index])
                    accuracy[i] += ber(
                        labels[test_index],
                        c.predict(data[np.ix_(best_features_indices, test_index)].T)
                    )

        return (features_selection.T * np.exp(accuracy)).T


class MeanWithClassifier(EmWithClassifier):
    def combine(self, features_selection, data, labels):
        return self.accuracy_fs(features_selection, data, labels).mean(axis=0)


class InfluenceWithClassifier(Influence, EmWithClassifier):
    def combine(self, features_selection, data, labels):
        return self.accuracy_fs(self._influence(features_selection), data, labels).mean(axis=0)


class MeanNormWithClassifier(MeanNormalizedSum, EmWithClassifier):
    def combine(self, features_selection, data, labels):
        return self.accuracy_fs(self._norm(features_selection), data, labels).mean(axis=0)
