from benchmarks import *
import numpy as np
import robustness_measure
import feature_selector
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score


class TestFeatureRanksGenerator:
    feature_selector = feature_selector.DummyFeatureSelector()

    def test_generate(self):
        data, labels = np.ones((10, 10)), np.arange(10)
        expected_results = np.array([
            np.arange(10),
            np.arange(10),
        ]).tolist()
        assert expected_results == self.feature_selector.generate(data, labels, KFold(n_splits=2).split(
            np.arange(labels.shape[0])), "rank").tolist()


class TestRobustnessBenchmark:
    benchmark = MeasureBenchmark(
        feature_selector=feature_selector.DummyFeatureSelector(),
        measure=robustness_measure.Dummy()
    )

    def test_robustness(self):
        assert 1 == self.benchmark.run(np.ones((10, 4)), np.arange(4), feature_selections=None)


class TestClassifierWrapper:
    classifier = ClassifierWrapper(
        DummyClassifier(strategy='constant', constant=1),
        accuracy_measure=f1_score
    )

    def test_run(self):
        data = np.random.randn(200, 10)
        labels = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, 1])

        train_index = np.arange(7)
        test_index = np.array([7, 8, 9])

        results = np.zeros((2, 2))
        result_index = (0, 1)

        expected_result = [
            [0, f1_score(labels[test_index], np.ones(3))],
            [0, 0]
        ]

        self.classifier.run_and_set_in_results(
            data,
            labels,
            train_index,
            test_index,
            results,
            result_index
        )

        assert np.allclose(expected_result, results.tolist())


class TestAccuracyBenchmark:
    benchmark = AccuracyBenchmark(
        feature_selector=feature_selector.DummyFeatureSelector(),
        classifiers=[
            DummyClassifier(strategy='constant', constant=1),
            DummyClassifier(strategy='constant', constant=-1),
        ]
    )

    def test_best_percent(self):
        l = np.arange(200)
        assert [199, 198] == AccuracyBenchmark.highest_percent(l, 0.01).tolist()


if __name__ == "__main__":
    test = TestAccuracyBenchmark()
    test.test_best_percent()
