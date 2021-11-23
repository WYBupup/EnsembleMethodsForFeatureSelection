from experiments import *
import numpy as np
import robustness_measure
import feature_selector
from sklearn.dummy import DummyClassifier


class TestRobustnessExperiment:
    experiment = MeasureExperiment(
        measures=robustness_measure.Dummy(),
        feature_selectors=[feature_selector.DummyFeatureSelector(), feature_selector.DummyFeatureSelector()]
    )

    def test_run(self):
        data = np.random.randn(200, 10)
        classes = np.array([1, 1, 1, 0, 0, 2, 0, 2, 0, 1])

        expected_results = [
            [1, 1]
        ]

        assert expected_results == self.experiment.run(data, classes).tolist()

    def test_print_results(self):
        experiment = MeasureExperiment(
            robustness_measure.Dummy(),
            [feature_selector.DummyFeatureSelector(), feature_selector.DummyFeatureSelector()]
        )
        experiment.results = np.array([[0.89, 0.1]])
        experiment.print_results()


class TestAccuracyExperiment:
    experiment = AccuracyExperiment(
        classifiers=DummyClassifier(strategy='constant', constant=1),
        feature_selectors=[feature_selector.DummyFeatureSelector(), feature_selector.DummyFeatureSelector()]
    )

    def test_run(self):
        data = np.random.randn(200, 10)
        classes = np.array([1, 1, 1, 0, 0, 2, 0, 2, 0, 1])

        expected_accuracy = [
            [4 / 10, 4 / 10],

        ]

        real = self.experiment.run(data, classes).tolist()
        print(real)
        self.experiment.print_results()
        assert expected_accuracy == real


if __name__ == "__main__":
    test = TestAccuracyExperiment()
    test.test_run()
