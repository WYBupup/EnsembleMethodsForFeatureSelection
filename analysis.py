from experiments import DataSetExperiment, DataSet_Precision_Experiment
from benchmarks import MeasureBenchmark, AccuracyBenchmark
from sklearn.neighbors import KNeighborsClassifier
from sklearn_utilities import SVC_Grid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
import robustness_measure
import goodness_measure
from feature_selector import DummyFeatureSelector

default_classifiers = [
    KNeighborsClassifier(3),
    SVC_Grid(kernel="linear"),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    LogisticRegressionCV(penalty='l1', solver='liblinear')
]


def run(data_sets, feature_selectors, jaccard_percentage=0.01, classifiers=None,
        measures=None, save=True, prefix=""):
    """
    :param data_sets: name list of data sets
    :param feature_selectors:  list of feature selectors
    :param jaccard_percentage:
    :param classifiers: list of classifiers
    :param measures:  list of robustness measure
    :param save: whether to save the result locally
    :param prefix:
    for every data sets:
        1: robustness measure vs feature selectors
        run robustness measurement for every feature selectors and types of robustness measurement
                                  feature selectors
        robustness measurements      mean+-std

        2: accuracy vs feature selectors
        run accuracy measurement for every feature selectors and types of classifiers
                                feature selectors
        classifiers                mean+-std

    if save
    save the robustness result   to ../results/RAW/prefix_jc10_robustness.npy
    (data set, feature selectors, measures, k(k-1)/2)

    save the accuracy result to  ../results/RAW/prefix_jc10_accuracy.npy
    (data set, feature selectors, classifiers, k)
    """
    if isinstance(data_sets, str):
        data_sets = [data_sets]

    if classifiers is None:
        classifiers = default_classifiers

    if measures is None:
        measures = [
            robustness_measure.JaccardIndex(percentage=jaccard_percentage)
        ]

    if len(prefix) > 0:
        prefix += "_"

    #robustness_exp = DataSetExperiment(
    #    MeasureBenchmark(measures),
    #    feature_selectors
    #)

    accuracy_exp = DataSetExperiment(
        AccuracyBenchmark(classifiers, percentage_of_features=jaccard_percentage),
        feature_selectors
    )

    precision_exp = DataSet_Precision_Experiment(feature_selectors)

    jcp = int(jaccard_percentage * 1e3)
    #robustness_exp.run(data_sets)
    #if save:
    #    robustness_exp.save_results(prefix + "jc{}_robustness".format(jcp))

    accuracy_exp.run(data_sets)
    if save:
        accuracy_exp.save_results(prefix + "jc{}_accuracy".format(jcp))

    precision_exp.run(data_sets)
    if save:
        precision_exp.save_results(prefix+" jc{}_precision".format(jcp))


def artificial(feature_selectors, jaccard_percentage=0.01, save=True, classifiers=None, measures=None):
    """
    For artificial data set:
    1. robustness vs feature selectors
                                       feature selectors
        jaccard index                     mean+-std
    2. rank precision (have feature labels) vs feature selectors
                                        feature selectors
            precisions                     mean+-std
    3. accuracy vs feature selectors
                                        feature selectors
            classifiers                   mean+-std
    ../results/RAW/artificial_jc10_robustness.npy (1, feature selectors, measures, k(k-1)/2)
    ../results/RAW/artificial_jc10_precision.npy (1, feature selectors, precisions, k)
    ../results/RAW/artificial_jc10_accuracy.npy (1, feature selectors, accuracy, k)
    """
    if classifiers is None:
        classifiers = default_classifiers
    if measures is None:
        measures = [
            robustness_measure.JaccardIndex(percentage=jaccard_percentage)
        ]

    # robustness_exp = DataSetExperiment(
    #     MeasureBenchmark(measures),
    #     feature_selectors
    # )

    precision_exp = DataSetExperiment(
        MeasureBenchmark([
            goodness_measure.Precision("artificial", 100),
            goodness_measure.Precision("artificial", 200),
            goodness_measure.Precision("artificial", 300),
            goodness_measure.Precision("artificial", 400),
            goodness_measure.Precision("artificial", 500),
            goodness_measure.Precision("artificial", 600),
            goodness_measure.Precision("artificial", 700),
            goodness_measure.Precision("artificial", 800),
            goodness_measure.Precision("artificial", 900),
            goodness_measure.Precision("artificial", 1000),
            goodness_measure.XPrecision("artificial"),
            goodness_measure.RankingLoss("artificial"),
            goodness_measure.LastRelevantFeature("artificial")
        ]),
        feature_selectors
    )

    # accuracy_exp = DataSetExperiment(
    #     AccuracyBenchmark(classifiers, percentage_of_features=jaccard_percentage),
    #     feature_selectors
    # )

    jcp = int(jaccard_percentage * 1e3)
    # robustness_exp.run("artificial")
    # if save:
    #     robustness_exp.save_results("artificial_jc{}_robustness".format(jcp))

    precision_exp.run("artificial")
    if save:
        precision_exp.save_results("artificial_jc{}_precision".format(jcp))

    # accuracy_exp.run("artificial")
    # if save:
    #     accuracy_exp.save_results("artificial_jc{}_accuracy".format(jcp))


def accuracy_with_all_features(data_sets, classifiers=None):
    """
    for every data sets:
        for every classifier:
            run accuracy measurement for every folds
                        dummy feature selectors(all features)
        classifiers            mean+-std
    (data sets, 1, classifier, k)
    and save the result and dimension information locally
    ../results/RAW/all_features_accuracy.npy
    ../results/RAW/all_features_accuracy_0.npy
    ../results/RAW/all_features_accuracy_1.npy
    ../results/RAW/all_features_accuracy_2.npy
    """
    if classifiers is None:
        classifiers = default_classifiers

    if isinstance(data_sets, str):
        data_sets = [data_sets]

    accuracy_exp = DataSetExperiment(
        AccuracyBenchmark(classifiers, percentage_of_features=100),
        DummyFeatureSelector()
    )

    accuracy_exp.run(data_sets)
    accuracy_exp.save_results("all_features_accuracy")
