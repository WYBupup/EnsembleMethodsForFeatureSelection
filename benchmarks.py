import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit
from abc import ABCMeta, abstractmethod
import multiprocessing
import ctypes
from feature_selector import FeatureSelector
from robustness_measure import Measure, JaccardIndex
from sklearn.base import clone as clone_classifier
from collections import Iterable
from accuracy_measure import ber

"""
MeasureBenchmark
For one feature selector's feature selections ranks(cv,D) use different robustness measurement to measure the result
if the feature selections are provided just use the ranks otherwise generate the feature selection use benchmark.cv() 
and self.feature_selector
MeasureBenchMark([robustness_measurement], feature_selector).run_raw_results(data, labels, feature_ranks)
                                                                                            ->(len(measures),k(k-1)/2)
MeasureBenchMark([robustness_measurement], feature_selector).run(data, labels, features_ranks) -> (len(measurement))



AccuracyBenchmark
For one feature selectors's rank (cv,D), assess its accuracy on top percentage features for different classifiers
if the features ranks is not provided, generate the ranks by self.feature selector and benchmark.cv()
AccuracyBenchmark([classifiers], feature_selector, percentage_feature_selected=0.01, accuracy_measurement=ber).run_raw_result(
                 data, labels, feature_ranks)-> (len(classifiers), k)
AccuracyBenchmark([classifiers], feature_selector, percentage_feature_selected=0.01, accuracy_measurement=ber).run(data, labels,
                feature_ranks) -> (len(classifiers))
                


FMeasureBenchmark
For one feature selector's ranks(cv,D) run the accuracy assessment for different classifiers-> (len(classifier))
and run robustness for Jaccard_index->(1)
and compute beta for different classifiers and mean the result
FMeasureBenchmark([classifiers], feature_selector, jaccard_percentage, beta).run(data, labels, robustness_feature_ranks, 
                    accuracy_feature_ranks)->(1)
"""


class Benchmark(metaclass=ABCMeta):
    feature_selector = None

    def generate_features_selection(self, data, labels):
        """
        :param data:  (D,N)
        :param labels: (N,)
        use self.feature_selector to rank features for every fold's training set  and return the ranks (cv,D)
        """
        if not isinstance(self.feature_selector, FeatureSelector):
            raise TypeError("feature_selector needs to be defined")

        return self.feature_selector.generate(data, labels, self.cv(labels.shape[0]), "rank")

    @staticmethod
    def cv(sample_size):
        pass

    # Returns mean results for each measure
    def run(self, *args, **kwargs):
        """
        args contain: data, labels, features_selection
        use every measurement in self.measures to measure the feature selection(ranks) which return len(self.measures)'s measure
        result while each result contains the combinations for all pairs of fold's result (k(k-1))/2(robust) or k(accuracy)
        return the average over the combinations return len(self.measures) measurement
        """
        mean_results = []
        for i, measure_results in enumerate(self.run_raw_result(*args, **kwargs)):
            mean_results.append(np.mean(measure_results))

        return np.array(mean_results)

    @abstractmethod
    # Returns an Iterable, one item per measures with all the results associated with it
    def run_raw_result(self, data, labels, features_selection=None) -> Iterable:
        pass

    @abstractmethod
    def get_measures(self):
        pass


class MeasureBenchmark(Benchmark):
    # robustness measure for single feature selector on particular data set with different robustness measurment
    def __init__(self, measure, feature_selector: FeatureSelector = None):
        """
        :param measure: list of class inherited from Measure in robustness_measure.py or one class
        :param feature_selector: Feature selector in feature_selector.py
        """
        self.feature_selector = feature_selector

        if not isinstance(measure, list):
            measure = [measure]

        for robustness_measure in measure:
            if not isinstance(robustness_measure, Measure):
                raise ValueError("At least one robustness measure does not inherit RobustnessMeasure")

        self.measures = measure

    def run_raw_result(self, data, labels, features_selection=None):
        """
        :param data: (D,N)
        :param labels: (N,)
        :param features_selection: the ranks of features for every fold's training set computed  by self.feature_selector
        (cv,D)

        if the feature selection is None , generate the feature selection
        use every measure method in self.measures to measure the feature selection and return the measures result
        return [result of measure[i] k*(k-1)/2  ] (len(measures))
        """
        if features_selection is None:
            features_selection = self.generate_features_selection(data, labels)

        features_selection = np.array(features_selection).T  # (D,CV)

        measures_results = multiprocessing.Manager().dict()

        processes = []
        for i in range(len(self.measures)):
            p = multiprocessing.Process(
                target=self.measures[i].run_and_set_in_results,
                kwargs={
                    'features_selection': features_selection,
                    'results': measures_results,
                    'result_index': i
                }
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # changed
        # to make the output in the order of different measurement
        return_ = [0 for _ in range(len(self.measures))]
        for i, measures_results in measures_results.items():
            return_[i] = measures_results
        return return_
        # return [measure_results for _, measure_results in measures_results.items()]

    @staticmethod
    def cv(sample_length):
        # TODO what cross validation should be used
        # the auythor used the outdated version
        cv_generator = KFold(n_splits=AccuracyBenchmark.n_fold, shuffle=True, random_state=1).split(np.arange(sample_length))
        cv_generator.__name__ = f"{AccuracyBenchmark.n_fold}Fold"
        return cv_generator
        # return ShuffleSplit(n_splits=10, test_size=0.1).split(np.arange(sample_length))

    def get_measures(self):
        return self.measures


class ClassifierWrapper:
    def __init__(self, classifier, accuracy_measure):
        """
        :param classifier:
        :param accuracy_measure: callable whose parameters are y_true and y_pred
        """
        self.classifier = classifier
        self.__name__ = type(classifier).__name__
        self.accuracy_measure = accuracy_measure

    def run_and_set_in_results(self, data, labels, train_index, test_index, results, result_index):
        """
        :param data: (D,N)
        :param labels: (N,)
        :param train_index:
        :param test_index:
        :param results:
        :param result_index:
        use training set to train self.classifier and use the trained classifier to predict the test set get y_pred and
        use self.accuracy_measure to measure the accuracy of test set and save the result in result[result_index]
        """
        np.random.seed()
        classifier = clone_classifier(self.classifier)

        classifier.fit(
            data[:, train_index].T,
            labels[train_index]
        )
        results[result_index] = self.accuracy_measure(
            labels[test_index],
            classifier.predict(data[:, test_index].T)
        )


class AccuracyBenchmark(Benchmark):
    # accuracy measurement for single feature selector on particular data set with different classifiers
    percentage_of_features = 0.01
    n_fold = 10

    def __init__(self, classifiers, feature_selector: FeatureSelector = None, percentage_of_features=None,
                 accuracy_measure=ber
                 ):
        """
        :param classifiers: list of classifiers
        :param feature_selector:  feature selector
        :param percentage_of_features:  percentage of features to select
        :param accuracy_measure: ber
        """
        self.feature_selector = feature_selector

        if percentage_of_features is not None:
            self.percentage_of_features = percentage_of_features

        if not isinstance(classifiers, list):
            classifiers = [classifiers]

        self.classifiers = [ClassifierWrapper(c, accuracy_measure) for c in classifiers]

    def run_raw_result(self, data, labels, features_selection=None):
        """
        :param data: (D,N)
        :param labels: (N,)
        :param features_selection:  (cv,D)
        if the feature selection is none generate the feature selection by self.feature_selector and self.cv (cv,D)
        For every classifier i in self.classifiers
            For every fold j in range(cv)
                select the top D*self.percentage_of_features ranked features and train the classifier i on fold j's
                training set and evaluate the classifier on fold j's test set
                return[i,j]=metric on test set
        ** it must be guaranteed that the folds for getting features ranking and the fold to measure accuracy is the same
        return result (len(self.classifier), cv)
        """
        if features_selection is None:
            features_selection = self.generate_features_selection(data, labels)

        features_indexes = {}
        for i, ranking in enumerate(features_selection):
            features_indexes[i] = self.highest_percent(ranking, self.percentage_of_features)

        # changed version
        # the original code cant save the result in multiprocessing
        shape = (len(self.classifiers), AccuracyBenchmark.n_fold)
        classification_accuracies = np.zeros(shape)
        # shared_array_base = multiprocessing.Array(ctypes.c_double, shape[0] * shape[1])
        # classification_accuracies = np.ctypeslib.as_array(shared_array_base.get_obj())
        # classification_accuracies = classification_accuracies.reshape(shape)
        measures_results = multiprocessing.Manager().dict()

        processes = []
        for i, classifier in enumerate(self.classifiers):
            for j, (train_index, test_index) in enumerate(self.cv(labels.shape[0])):
                p = multiprocessing.Process(
                    target=classifier.run_and_set_in_results,
                    kwargs={
                        'data': data[features_indexes[j], :],
                        'labels': labels,
                        'train_index': train_index,
                        'test_index': test_index,
                        'results': measures_results,  # classification_accuracies
                        'result_index': i * shape[1] + j  # (i, j)
                    }
                )
                p.start()
                processes.append(p)

        for p in processes:
            p.join()
        for i, temp in measures_results.items():
            classification_accuracies[int(i / shape[1]), int(i % shape[1])] = temp

        return classification_accuracies

    @staticmethod
    def cv(sample_length):
        # TODO what cross validation should be used
        # the auythor used the outdated version
        cv_generator = KFold(n_splits=AccuracyBenchmark.n_fold, shuffle=True, random_state=1).split(np.arange(sample_length))
        cv_generator.__name__ = f"{AccuracyBenchmark.n_fold}Fold"
        return cv_generator
        # return ShuffleSplit(n_splits=10, test_size=0.1).split(np.arange(sample_length))

    # 1% best features
    @staticmethod
    def highest_percent(features_selection, percentage):
        """
        :param features_selection: rank of features (D,)
        :param percentage:
        return the indices of top D*percentage ranked features
        """
        if percentage == 100:
            return np.arange(features_selection.size)
        size = 1 + int(features_selection.size * percentage)
        return np.argsort(features_selection)[:-size:-1]

    def get_measures(self):
        return self.classifiers


class FMeasureBenchmark:
    # average F measurement (robust: Jaccard index,  accuracy different classifier) over different classifiers
    # for single feature selector
    def __init__(self, classifiers, feature_selector: FeatureSelector = None, jaccard_percentage=0.01, beta=1):
        self.robustness_benchmark = MeasureBenchmark(
            [JaccardIndex(percentage=jaccard_percentage)],
            feature_selector=feature_selector
        )
        self.accuracy_benchmark = AccuracyBenchmark(
            classifiers,
            feature_selector=feature_selector,
            percentage_of_features=jaccard_percentage
        )
        self.beta = beta

    def run(self, data, labels, robustness_features_selection=None, accuracy_features_selection=None):
        """
        :param data: (D,N)
        :param labels: (N,)
        :param robustness_features_selection: (cv,D)
        :param accuracy_features_selection: (cv,D)
        analyze the jaccard index for evey combinations of different fold's feature selection and average it which get
        the robustness(1,)
        analyze the accuracy for every classifier as described in AccuracyBenchmark which get the accuracy (classifier)
        compute F score  for classifier
        return the mean F score over every classifier (1,)
        """
        return np.mean(self.f_measure(
            self.robustness_benchmark.run(data, labels, robustness_features_selection),  # (1,)
            self.accuracy_benchmark.run(data, labels, accuracy_features_selection),  # (classifier,)
            self.beta
        ))

    @staticmethod
    def f_measure(robustness, accuracy, beta=1):
        return ((beta ** 2 + 1) * robustness * accuracy) / (beta ** 2 * robustness + accuracy)
