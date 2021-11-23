from feature_selector import SymmetricalUncertainty, Relief, SVM_RFE, LassoFeatureSelector, Random, FeatureSelector
import ensemble_methods
import analysis
import artificial_data
import numpy as np
import matplotlib.pyplot as plt
from data_sets import DataSets
import itertools
from experiments import Experiment
from tabulate import tabulate
from robustness_measure import JaccardIndex
from sklearn_utilities import SVC_Grid
import ensemble_homogeneous
import benchmarks
import warnings
import sys
from analyse_weights import AnalyseBenchmarkResults

warnings.filterwarnings('ignore')


# GENERATION OF DATA SET

# total_features = 1e4
# n_significant_features = 100
# DataSets.save_artificial(
#     *artificial_data.generate(
#         n_samples=300,
#         n_features=total_features,
#         n_significant_features=n_significant_features,
#         feature_distribution=artificial_data.multiple_distribution(
#             distributions=[
#                 artificial_data.multivariate_normal(
#                     mean=artificial_data.constant(0),
#                     cov=artificial_data.uniform(0, 1)
#                 ),
#                 artificial_data.normal(0, 1)
#             ],
#             shares=[0.5, 0.5]
#         ),
#         insignificant_feature_distribution=artificial_data.multiple_distribution(
#             distributions=[
#                 artificial_data.multivariate_normal(
#                     mean=artificial_data.constant(0),
#                     cov=artificial_data.uniform(0, 1)
#                 ),
#                 artificial_data.normal(0, 1)
#             ],
#             shares=[0.5, 0.5]
#         ),
#         labeling=artificial_data.linear_labeling(weights=np.ones(n_significant_features))
#     )
# )
#
# data, _ = DataSets.load("artificial")
# cov = np.cov(data[:200])
# plt.imshow(cov, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.show()
# plt.clf()


def compare_feature_selectors(p):
    """
    :param p:  the percentage of top ranked features to select
    compare  following selectors with respect to robustness, accuracy
Relief, SVM-RFE, SymmetricalUncertainty, Lasso, Random, Ensemble_mean, Ensemble_normalized_mean, Ensemble_classifier_mean
    for different data set

    For every data set
    compute
    1.robustness
                          feature selectors
    Jaccard index             mean+-std

    2. accuracy
                          feature selectors
    classifier                 mean+-std

    3. only for artificial data set
                         feature selectors
    rank precision            mean+-std

    and save the corresponding experiment result
    """
    feature_selectors = [
        SymmetricalUncertainty(),
        Relief(),
        SVM_RFE(percentage_features_to_select=p),
    ]

    # e_methods = [
    #     ensemble_methods.Mean(data_set_feature_selectors=feature_selectors),
    #     ensemble_methods.MeanNormalizedSum(data_set_feature_selectors=feature_selectors),
    #     ensemble_methods.MeanWithClassifier(data_set_feature_selectors=feature_selectors,
    #                                         classifiers=analysis.default_classifiers),
    # ]

    e_methods = [
        ensemble_homogeneous.Mean(data_set_feature_selector=SVM_RFE(percentage_features_to_select=p)),
        ensemble_homogeneous.Mean(data_set_feature_selector=Relief()),
        ensemble_homogeneous.Mean(data_set_feature_selector=SymmetricalUncertainty()),
        ensemble_homogeneous.Mean(data_set_feature_selector=LassoFeatureSelector()),
        ensemble_methods.Mean(data_set_feature_selectors=feature_selectors),
        ensemble_methods.Mean(data_set_feature_selectors=feature_selectors + [LassoFeatureSelector()]),
    ]

    fs = feature_selectors + [LassoFeatureSelector(), Random()] + e_methods

    data_sets = ["artificial", "colon", "arcene", "dexter", "gisette", "dorothea", ]  #
    # data_sets = ["artificial"]
    # # analysis.accuracy_with_all_features(data_sets=data_sets, classifiers=[SVC_Grid(kernel="linear")])
    # analysis.artificial(fs, jaccard_percentage=p,
    #                     measures=[JaccardIndex(percentage=0.01), JaccardIndex(percentage=0.05),
    #                               JaccardIndex(percentage=0.1)],
    #                     classifiers=[SVC_Grid(kernel="linear")])

    # analysis.run(data_sets, fs, jaccard_percentage=p,
    #              measures=[JaccardIndex(percentage=0.01), JaccardIndex(percentage=0.05),
    #                        JaccardIndex(percentage=0.1)], classifiers=[SVC_Grid(kernel="linear")], prefix="data_sets")
    AnalyseBenchmarkResults(fs).run("artificial", save_to_file=False)


def combinations():
    """
    experiment all the combinations of [symmetricalUncertainty, Relief, SVM-RFE, Lasso] in terms of accuracy and
    robustness (1 fs, 2 fs, 3 fs, 4 fs)
    for data set ["artificial", "colon", "arcene", "dexter", "gisette"]

    For every data set:
    1. robustness
                                   feature selectors
    Jaccard index                    mean+-std

    2. accuracy
                                   feature selectors
    Classifiers                        mean+-std

    and save the experiment results
    """
    fs = [
        SymmetricalUncertainty(),
        Relief(),
        SVM_RFE(),
        LassoFeatureSelector(),
    ]

    e_methods = [
        ensemble_methods.Mean(data_set_feature_selectors=fs),
        ensemble_methods.Influence(data_set_feature_selectors=fs),
        ensemble_methods.MeanNormalizedSum(data_set_feature_selectors=fs),
        ensemble_methods.MeanWithClassifier(
            data_set_feature_selectors=fs,
            classifiers=analysis.default_classifiers
        ),
        ensemble_methods.InfluenceWithClassifier(
            data_set_feature_selectors=fs,
            classifiers=analysis.default_classifiers
        ),
        ensemble_methods.MeanNormWithClassifier(
            data_set_feature_selectors=fs,
            classifiers=analysis.default_classifiers
        ),
    ]

    for comb in itertools.combinations(list(range(4)), 3):
        comb_fs = [fs[i] for i in comb]
        e_methods.extend([
            ensemble_methods.Mean(data_set_feature_selectors=comb_fs),
            ensemble_methods.Influence(data_set_feature_selectors=comb_fs),
            ensemble_methods.MeanNormalizedSum(data_set_feature_selectors=comb_fs),
            ensemble_methods.MeanWithClassifier(
                data_set_feature_selectors=comb_fs,
                classifiers=analysis.default_classifiers
            ),
            ensemble_methods.InfluenceWithClassifier(
                data_set_feature_selectors=comb_fs,
                classifiers=analysis.default_classifiers
            ),
            ensemble_methods.MeanNormWithClassifier(
                data_set_feature_selectors=comb_fs,
                classifiers=analysis.default_classifiers
            ),
        ])

    for comb in itertools.combinations(list(range(4)), 2):
        comb_fs = [fs[i] for i in comb]
        e_methods.extend([
            ensemble_methods.Mean(data_set_feature_selectors=comb_fs),
            ensemble_methods.Influence(data_set_feature_selectors=comb_fs),
            ensemble_methods.MeanNormalizedSum(data_set_feature_selectors=comb_fs),
            ensemble_methods.MeanWithClassifier(
                data_set_feature_selectors=comb_fs,
                classifiers=analysis.default_classifiers
            ),
            ensemble_methods.InfluenceWithClassifier(
                data_set_feature_selectors=comb_fs,
                classifiers=analysis.default_classifiers
            ),
            ensemble_methods.MeanNormWithClassifier(
                data_set_feature_selectors=comb_fs,
                classifiers=analysis.default_classifiers
            ),
        ])

    data_sets = ["artificial", "colon", "arcene", "dexter", "gisette"]

    analysis.run(data_sets, fs + e_methods, prefix="combinations")


def combination_plot():
    """
    analyze the result of feature selectors for all combinatiosn of single feature selectors
    """
    # (data set, feature selectors, classifiers, k)
    accuracy = np.load('../results/RAW/combinations_jc10_accuracy.npy')
    # (data set, feature selectors, 1, k(k-1)/2)
    robustness = np.load('../results/RAW/combinations_jc10_robustness.npy')
    labels = []  # names of feature_selectors
    with open('../results/RAW/combinations_jc10_accuracy_1.txt') as f:
        for label in f:
            labels.append(label.strip())

    #  (data sets, feature selectors, classifiers)
    beta = 2 * accuracy.mean(axis=-1) * robustness.mean(axis=-1) / (robustness.mean(axis=-1) + accuracy.mean(axis=-1))
    #  (data sets, feature selectors, 1, classifiers)
    beta = beta.reshape(4, 70, 1, beta.shape[-1])

    methods_label = [
        "SU",
        "RLF",
        "SVM",
        "LSO",
        "Avg",
        "Inf",
        "AvgNormed",
        "Avg_C",
        "Inf_C",
        "AvgNormed_C",
    ]

    combinations_label = [
        "SU",
        "RLF",
        "SVM",
        "LSO",
        "SU+RLF+SVM+LSO",
        "SU+RLF+SVM",
        "SU+RLF+LSO",
        "SU+SVM+LSO",
        "RLF+SVM+LSO",
        "SU+RLF",
        "SU+SVM",
        "SU+LSO",
        "RLF+SVM",
        "RLF+LSO",
        "SVM+LSO"
    ]

    data = [
        accuracy,
        robustness,
        beta
    ]

    #  (3,feature selectors)
    mean = list(map(
        lambda a: a.mean(axis=(0, -2, -1)),
        data
    ))

    #  (3,feature selectors)
    mean_error = list(map(
        lambda a: a.std(axis=(-1)).mean(axis=(0, -1)),
        data
    ))

    # [accuracy for different combinations of single feature selectors(15),
    #   robustness for different combinations of single feature selectors(15),
    #   F score for different combination of single feature selectors(15)]
    comb = list(map(
        lambda a: np.hstack((
            a[:, :4].mean(axis=(0, -2, -1)),
            a[:, 4:].reshape(4, 11, 6, a.shape[-2], a.shape[-1]).mean(axis=(0, -3, -2, -1))
        )),
        data
    ))

    # [accuracy for different combinations of single feature selectors(15),
    #   robustness for different combinations of single feature selectors(15),
    #   F score for different combination of single feature selectors(15)]
    comb_error = list(map(
        lambda a: np.hstack((
            a[:, :4].std(axis=(-1)).mean(axis=(0, -1)),
            a[:, 4:].reshape(4, 11, 6, a.shape[-2], a.shape[-1]).std(axis=(-1)).mean(axis=(0, -2, -1))
        )),
        data
    ))

    meth = list(map(
        lambda a: np.hstack((
            a[:, :4].mean(axis=(0, -2, -1)),
            a[:, 4:].reshape(4, 11, 6, a.shape[-2], a.shape[-1])[:, [1, 5, 6, 8]].mean(axis=(0, 1, -2, -1))
        )),
        data
    ))

    meth_error = list(map(
        lambda a: np.hstack((
            a[:, :4].std(axis=(-1)).mean(axis=(0, -1)),
            a[:, 4:].reshape(4, 11, 6, a.shape[-2], a.shape[-1])[:, [1, 5, 6, 8]].std(axis=(-1)).mean(axis=(0, 1, -1))
        )),
        data
    ))

    def sprint(order, mean, std, header):
        rows = [
            ["accuracy"] + list(
                map(lambda m, s: "{:.2%} ± {:.2%}".format(m, s), mean[0][order].tolist(), std[0][order].tolist())),
            ["robustness"] + list(
                map(lambda m, s: "{:.2%} ± {:.2%}".format(m, s), mean[1][order].tolist(), std[1][order].tolist())),
            ["beta"] + list(
                map(lambda m, s: "{:.2%} ± {:.2%}".format(m, s), mean[2][order].tolist(), std[2][order].tolist())),
        ]
        print(tabulate(rows, ["Measure"] + [header[i] for i in order], tablefmt='pipe'))
        print()

    def table(mean, std, header):
        order = list(map(
            lambda a: np.argsort(a)[::-1],
            mean
        ))

        print("SORTED BY ACCURACY")
        sprint(order[0], mean, std, header)
        print("SORTED BY ROBUSTNESS")
        sprint(order[1], mean, std, header)
        print("SORTED BY BETA")
        sprint(order[2], mean, std, header)

        x = np.arange(mean[0].shape[0])
        for i in range(3):
            plt.figure(num=["accuracy", "robustness", "beta"][i])
            plt.errorbar(x, mean[i][order[i]], yerr=std[i][order[i]])
            plt.xticks(x, [header[k] for k in order[i]], rotation=-90)
            axes = plt.gca()
            axes.set_xlim([-1, x.size])
            vals = axes.get_yticks()
            axes.set_yticklabels(['{:3.2f}%'.format(x * 100) for x in vals])
            plt.tight_layout()

    # print("DATA\n")
    # table(mean, mean_error, labels)
    # print("\nCOMBINATIONS\n")
    # table(comb, comb_error, combinations_label)
    print("\nMETHODS\n")
    table(meth, meth_error, methods_label)

    plt.figure()
    plt.hist(mean[0], label="accuracy")
    plt.hist(mean[1], bins=20, label="robustness")
    plt.hist(mean[2], bins=20, label="beta")
    plt.legend()
    plt.tight_layout()
    plt.show()


# added

# TODO show figures of error bar
def show_error_bar(file_name, accuracy_all_features_file_name):
    accuracy_all_features = np.load(accuracy_all_features_file_name)  # (data_sets, 1, classifiers, k)
    accuracy_result = np.load(file_name + "_accuracy.npy")  # (data_set, fs, classifiers, k)
    robustness_result = np.load(file_name + "_robustness.npy")  # (data set, fs, measures, k(k-1)/2)
    accuracy_result = np.mean(accuracy_result, axis=-2)  # mean over different classifiers
    robustness_result = np.mean(robustness_result, axis=-2)  # jaccard 1%
    data_set_names = []
    with open(file_name + "_accuracy_0.txt", "r") as f:
        for line in f.readlines():
            data_set_names.append(line.strip())
    feature_selectors_names = []
    with open(file_name + "_accuracy_1.txt", "r") as f:
        for line in f.readlines():
            feature_selectors_names.append(line.strip())

    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan", "midnightblue"]
    for i in range(len(data_set_names)):
        accuracy_all_features_temp = np.mean(accuracy_all_features[i, 0], axis=0)
        fig1, axe1 = plt.subplots(1, 1)
        fig1.set_size_inches(10, 10)
        # axe1.set_title(f"[{data_set_names[i]}] single feature selector vs homogeneous ensemble")
        axe1.set_title(f"[{data_set_names[i]}] single feature selector")
        axe1.set_xlabel("accuracy")
        axe1.set_ylabel("robustness")
        axe1.set_xlim(0, 1)
        axe1.set_ylim(0, 1)
        axe1.fill_between(np.arange(np.min(accuracy_all_features_temp), np.max(accuracy_all_features_temp), 0.01), y1=0,
                          y2=1, color="bisque", alpha=0.5)

        fig2, axe2 = plt.subplots(1, 1)
        fig2.set_size_inches(10, 10)
        # axe2.set_title(f"[{data_set_names[i]}]single feature selector vs heterogeneous ensemble")
        axe2.set_title(f"[{data_set_names[i]}]single feature selector")
        axe2.set_xlabel("accuracy")
        axe2.set_ylabel("robustness")
        axe2.set_xlim(0, 1)
        axe2.set_ylim(0, 1)
        axe2.fill_between(np.arange(np.min(accuracy_all_features_temp), np.max(accuracy_all_features_temp), 0.01), y1=0,
                          y2=1, color="bisque", alpha=0.5)

        for j in range(len(feature_selectors_names)):
            if feature_selectors_names[j].startswith("Heterogeneous")or \
                    feature_selectors_names[j].startswith("Homogeneous") or\
                    feature_selectors_names[j] == "Heterogeneous_Mean_SU_RLF_SVM_LSO" or \
                    feature_selectors_names[j] == "LassoFeatureSelector" or \
                    feature_selectors_names[j] == "Homogeneous_Mean_LSO":
                continue
            if j <= 8:
                axe1.errorbar(x=np.mean(accuracy_result[i, j]), y=np.mean(robustness_result[i, j]),
                              yerr=np.std(robustness_result[i, j]), xerr=np.std(accuracy_result[i, j]),
                              ecolor=colors[j], label=feature_selectors_names[j], fmt='o', color=colors[j])
            if j <= 4 or j > 8:
                axe2.errorbar(x=np.mean(accuracy_result[i, j]), y=np.mean(robustness_result[i, j]),
                              yerr=np.std(robustness_result[i, j]), xerr=np.std(accuracy_result[i, j]),
                              ecolor=colors[j], label=feature_selectors_names[j], fmt='o', color=colors[j])

        axe1.legend(loc="best")
        axe2.legend(loc="best")
        plt.show()
        # fig1.savefig(file_name + data_set_names[i] + "_homogeneous.png")
        # fig2.savefig(file_name + data_set_names[i] + "_heterogeneous.png")
        fig1.savefig(file_name + data_set_names[i] + ".png")


def Fmeasurement(accuracy_file_name, robustness_file_name, data_sets_file, feature_selectors_file):
    accuracy = np.load(accuracy_file_name)  # (data sets, fs, classifiers, cv)
    robustness = np.load(robustness_file_name)  # (data sets, fs, robustness measurement, cv)
    accuracy = np.mean(accuracy, axis=(-1, -2))
    robustness = np.mean(robustness, axis=(-1, -2))
    F = accuracy * robustness / (accuracy + robustness)  # (data sets, fs)
    data_sets = []
    feature_selectors = []
    with open(data_sets_file, "r") as f:
        for line in f.readlines():
            data_sets.append(line.strip())
    with open(feature_selectors_file, "r") as f:
        for line in f.readlines():
            feature_selectors.append(line.strip())

    # show table
    rows = [["F score"] + feature_selectors]
    for i in range(F.shape[0]):
        row = [data_sets[i]]
        row += map(lambda x: "{:.2}".format(x), F[i].tolist())
        rows.append(row)
    mean = np.mean(F, axis=0)
    row = ["mean"]
    row += map(lambda x: "{:.2}".format(x), mean.tolist())
    rows.append(row)
    print(tabulate(rows[1:], rows[0], tablefmt="pipe"))
    return rows


# # generation of my data set
# total_features = 10000
# n_significant_features = 100
# DataSets.save_artificial(
#     *artificial_data.generate(
#         n_samples=300,
#         n_features=total_features,
#         n_significant_features=n_significant_features,
#         feature_distribution=artificial_data.multiple_distribution(
#             distributions=[artificial_data.multivariate_normal(
#                 mean=artificial_data.constant(0),
#                 cov=artificial_data.uniform(0, 1)
#             ), artificial_data.normal(0, 1)],
#             shares=[0.5, 0.5]
#         ),
#         insignificant_feature_distribution=artificial_data.uniform(0, 1),
#         labeling=artificial_data.linear_labeling(weights=np.ones(n_significant_features))
#     )
# )
#
# data, labels = DataSets.load("artificial")
# fig, axes = plt.subplots(1, 2)
# cov = np.cov(data)
# axes[0].imshow(cov, cmap='hot', interpolation='nearest')
# axes[0].set_title("all features")
# cov = np.cov(data[:200])
# axes[1].imshow(cov, cmap="hot", interpolation="nearest")
# axes[1].set_title("200 features")
# # plt.colorbar()
# plt.show()

if __name__ == "__main__":
    # feature_selectors = [
    #     SymmetricalUncertainty(),
    #     Relief(),
    #     SVM_RFE(percentage_features_to_select=0.01),
    # ]
    # e_methods = [
    #     ensemble_homogeneous.Mean(data_set_feature_selector=SVM_RFE(percentage_features_to_select=0.01)),
    #     ensemble_homogeneous.Mean(data_set_feature_selector=Relief()),
    #     ensemble_homogeneous.Mean(data_set_feature_selector=SymmetricalUncertainty()),
    #     ensemble_homogeneous.Mean(data_set_feature_selector=LassoFeatureSelector()),
    #     ensemble_methods.Mean(data_set_feature_selectors=feature_selectors),
    #     ensemble_methods.Mean(data_set_feature_selectors=feature_selectors + [LassoFeatureSelector()]),
    # ]
    # for fs in e_methods:
    #     weights = fs.weight_data_set("artificial", benchmarks.MeasureBenchmark.cv)
    # compare_feature_selectors(0.01)
    show_error_bar(r"D:\machine_learning\feature selection\experiment\results\RAW\artificial_jc10",
                   r"D:\machine_learning\feature selection\experiment\results\RAW\all_features_accuracy.npy")
    # log_print = open('precision.log', 'w')
    # sys.stdout = log_print
    # sys.stderr = log_print
    # Fmeasurement(r"D:\machine_learning\feature selection\experiment\results_v1\RAW\data_sets_jc10_accuracy.npy",
    #              r"D:\machine_learning\feature selection\experiment\results_v1\RAW\data_sets_jc10_robustness.npy",
    #              r"D:\machine_learning\feature selection\experiment\results_v1\RAW\data_sets_jc10_accuracy_0.txt",
    #              r"D:\machine_learning\feature selection\experiment\results_v1\RAW\data_sets_jc10_accuracy_1.txt")


