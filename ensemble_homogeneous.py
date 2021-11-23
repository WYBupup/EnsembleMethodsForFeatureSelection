# encoding:utf-8
from feature_selector import DataSetFeatureSelector, FeatureSelector
from abc import ABCMeta, abstractmethod
import multiprocessing
from data_sets import PreComputedData, DataSets
from io_utils import mkdir
import numpy as np
from sklearn.utils import resample


class Homogeneous_EnsembleMethod(DataSetFeatureSelector, metaclass=ABCMeta):
    max_parallelism = multiprocessing.cpu_count()

    def __init__(self, data_set_feature_selector: FeatureSelector):
        super().__init__()

        if not isinstance(data_set_feature_selector, FeatureSelector):
            raise ValueError(" Only FeatureSelector in feature_selector.py can be used")

        self.feature_selector = data_set_feature_selector
        self.__name__ = "Homogeneous_" + self.__name__ + "_" + str(self.feature_selector)

    def rank_data_set(self, data_set, cv_generator):
        super().rank_data_set(data_set, cv_generator)

        data, labels = DataSets.load(data_set)
        cv = cv_generator(labels.shape[0])

        try:
            return PreComputedData.load(data_set, cv, "rank", self)
        except FileNotFoundError:
            print(f"=>Generating features ranks of {data_set} ({cv.__name__}) with {self.__name__}")
            try:
                cv_indices = PreComputedData.load_cv(data_set, cv)
            except FileNotFoundError:
                mkdir(PreComputedData.cv_dir(data_set, cv))

                cv_indices = list(cv)
                np.save(PreComputedData.cv_file_name(data_set, cv), cv_indices)

            bench_features_selection = []  # (cv, cv_ensemble, D)
            for i in range(len(cv_indices)):
                data_ = data[:, cv_indices[i][0]]
                labels_ = labels[cv_indices[i][0]]
                weights_cv = self.feature_selector.generate(data_, labels_,
                                                   self.bootstrap(labels_.shape[0]),
                                                   "weight") # (40,D)
                print(f"generate weights for cv{i} shape:{weights_cv.shape}")
                bench_features_selection.append(
                    weights_cv
                )
            bench_features_selection = np.array(bench_features_selection)  # (cv, cv_ensemble, D)

            feature_selection = multiprocessing.Manager().dict()

            with multiprocessing.Pool(processes=self.max_parallelism) as pool:
                for i in range(bench_features_selection.shape[0]):
                    pool.apply_async(
                        self.run_and_set_in_result,
                        kwds={
                            "results": feature_selection,
                            "result_index": i,
                            "feature_selection": bench_features_selection[i],
                            "data": data[:, cv_indices[i][0]],
                            "labels": labels[cv_indices[i][0]]
                        }
                    )
                pool.close()
                pool.join()
            ranks = [0 for _ in range(bench_features_selection.shape[0])]
            for i, rank in feature_selection.items():
                ranks[i] = rank

            ranks = np.array(ranks)
            self.__save(data_set, cv, "rank", ranks)
            return ranks

    def weight_data_set(self, data_set, cv_generator):
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

    def run_and_set_in_result(self, results, result_index, feature_selection, data, labels):
        np.random.seed()
        results[result_index] = self.rank_weights(self.combine(feature_selection, data, labels))

    def __save(self, data_set, cv, method, feature_selection):
        mkdir(PreComputedData.dir_name(data_set, cv, method))
        np.save(PreComputedData.file_name(data_set, cv, method, self), feature_selection)

    @staticmethod
    def bootstrap(samples):
        # re_sample 40 bags
        cv_indices = []
        indices = np.arange(samples)
        for _ in range(40):
            cv_indices.append((resample(indices), 0))
        return cv_indices

    @abstractmethod
    def combine(self, feature_selection, data, labels):
        pass


class Mean(Homogeneous_EnsembleMethod):
    def combine(self, feature_selection, data, labels):
        return feature_selection.mean(axis=0)
