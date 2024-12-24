import unittest

import numpy as np
import pytest

from CPDShell.Core.algorithms.bayesian_algorithm import BayesianAlgorithm
from CPDShell.Core.algorithms.BayesianCPD.detectors.simple_detector import SimpleDetector
from CPDShell.Core.algorithms.BayesianCPD.hazards.constant_hazard import ConstantHazard
from CPDShell.Core.algorithms.BayesianCPD.likelihoods.gaussian_unknown_mean_and_variance import (
    GaussianUnknownMeanAndVariance,
)
from CPDShell.Core.algorithms.BayesianCPD.localizers.simple_localizer import SimpleLocalizer


class TestBayesianAlgorithm(unittest.TestCase):
    def test_consecutive_detection(self):
        np.random.seed(1)

        num_of_tests = 10
        size = 500
        change_point = 250

        learning_steps = 50
        likelihood = GaussianUnknownMeanAndVariance()
        hazard = ConstantHazard(rate=1.0 / (1.0 - 0.5 ** (1.0 / 500)))
        detector = SimpleDetector(threshold=0.04)
        localizer = SimpleLocalizer()

        bayesian_algorithm = BayesianAlgorithm(
            learning_steps=learning_steps, likelihood=likelihood, hazard=hazard, detector=detector, localizer=localizer
        )

        for data_num in range(num_of_tests):
            data = np.concatenate(
                (
                    np.random.normal(loc=0, scale=1, size=change_point),
                    np.random.normal(loc=5, scale=2, size=size - change_point),
                )
            )

            result = bayesian_algorithm.detect(data)

            self.assertTrue(result, msg="There was undetected change point in data")

    def test_correctness_of_consecutive_detection(self):
        np.random.seed(1)

        num_of_tests = 10
        size = 500
        change_point = 250

        learning_steps = 50
        outer_likelihood = GaussianUnknownMeanAndVariance()
        outer_hazard = ConstantHazard(rate=1.0 / (1.0 - 0.5 ** (1.0 / 500)))
        outer_detector = SimpleDetector(threshold=0.04)
        outer_localizer = SimpleLocalizer()

        outer_bayesian_algorithm = BayesianAlgorithm(
            learning_steps=learning_steps,
            likelihood=outer_likelihood,
            hazard=outer_hazard,
            detector=outer_detector,
            localizer=outer_localizer,
        )

        for data_num in range(num_of_tests):
            data = np.concatenate(
                (
                    np.random.normal(loc=0, scale=1, size=change_point),
                    np.random.normal(loc=5, scale=2, size=size - change_point),
                )
            )

            learning_steps = 50
            inner_likelihood = GaussianUnknownMeanAndVariance()
            inner_hazard = ConstantHazard(rate=1.0 / (1.0 - 0.5 ** (1.0 / 500)))
            inner_detector = SimpleDetector(threshold=0.04)
            inner_localizer = SimpleLocalizer()

            inner_bayesian_algorithm = BayesianAlgorithm(
                learning_steps=learning_steps,
                likelihood=inner_likelihood,
                hazard=inner_hazard,
                detector=inner_detector,
                localizer=inner_localizer,
            )

            outer_result = outer_bayesian_algorithm.detect(data)
            inner_result = inner_bayesian_algorithm.detect(data)

            self.assertTrue(
                outer_result == inner_result, msg="Consecutive and independent detection should give same results"
            )

    def test_consecutive_localization(self):
        np.random.seed(1)

        num_of_tests = 10
        size = 500
        change_point = 250
        tolerable_deviation = 25

        learning_steps = 50
        likelihood = GaussianUnknownMeanAndVariance()
        hazard = ConstantHazard(rate=1.0 / (1.0 - 0.5 ** (1.0 / 500)))
        detector = SimpleDetector(threshold=0.04)
        localizer = SimpleLocalizer()

        bayesian_algorithm = BayesianAlgorithm(
            learning_steps=learning_steps, likelihood=likelihood, hazard=hazard, detector=detector, localizer=localizer
        )

        for data_num in range(num_of_tests):
            data = np.concatenate(
                (
                    np.random.normal(loc=0, scale=1, size=change_point),
                    np.random.normal(loc=5, scale=2, size=size - change_point),
                )
            )

            result = bayesian_algorithm.localize(data)

            self.assertTrue(
                len(result) > 0
                and change_point - tolerable_deviation <= result[0] <= change_point + tolerable_deviation,
                msg=f"Incorrect change point for data {data_num + 1}",
            )

    def test_correctness_of_consecutive_localization(self):
        np.random.seed(1)

        num_of_tests = 10
        size = 500
        change_point = 250

        learning_steps = 50
        outer_likelihood = GaussianUnknownMeanAndVariance()
        outer_hazard = ConstantHazard(rate=1.0 / (1.0 - 0.5 ** (1.0 / 500)))
        outer_detector = SimpleDetector(threshold=0.04)
        outer_localizer = SimpleLocalizer()

        outer_bayesian_algorithm = BayesianAlgorithm(
            learning_steps=learning_steps,
            likelihood=outer_likelihood,
            hazard=outer_hazard,
            detector=outer_detector,
            localizer=outer_localizer,
        )

        for data_num in range(num_of_tests):
            data = np.concatenate(
                (
                    np.random.normal(loc=0, scale=1, size=change_point),
                    np.random.normal(loc=5, scale=2, size=size - change_point),
                )
            )

            learning_steps = 50
            inner_likelihood = GaussianUnknownMeanAndVariance()
            inner_hazard = ConstantHazard(rate=1.0 / (1.0 - 0.5 ** (1.0 / 500)))
            inner_detector = SimpleDetector(threshold=0.04)
            inner_localizer = SimpleLocalizer()

            inner_bayesian_algorithm = BayesianAlgorithm(
                learning_steps=learning_steps,
                likelihood=inner_likelihood,
                hazard=inner_hazard,
                detector=inner_detector,
                localizer=inner_localizer,
            )

            outer_result = outer_bayesian_algorithm.localize(data)
            inner_result = inner_bayesian_algorithm.localize(data)

            self.assertTrue(
                outer_result == inner_result, msg="Consecutive and independent localization should give same results"
            )


class TestConstantHazard:
    @pytest.mark.parametrize("hazard_rate,max_run_length", [(1.1, 50), (10, 100), (200, 250), (500.325251, 500)])
    def test_for_constants(self, hazard_rate, max_run_length):
        constant_hazard = ConstantHazard(hazard_rate)
        run_lengths = np.arange(max_run_length)
        hazard_probs = constant_hazard.hazard(run_lengths)

        assert hazard_probs.shape[0] == max_run_length, (
            f"There must be {max_run_length} probabilities;" f"got {hazard_probs.shape[0]} instead"
        )
        assert np.all(hazard_probs == 1 / hazard_rate), f"Hazard probabilities must be {1 / hazard_rate}"


class TestGaussianUnknownMeanAndVariance(unittest.TestCase):
    def test_learning_and_update(self):
        np.random.seed(1)

        num_of_tests = 10
        size = 500
        change_point = 250
        learning_steps = 50

        time_after_learning = 51
        threshold_after_learning = 0.01

        time_before_change_point = 249
        threshold_change_point = 0.05

        gaussian_likelihood = GaussianUnknownMeanAndVariance()

        for data_num in range(num_of_tests):
            data = np.concatenate(
                (
                    np.random.normal(loc=0, scale=1, size=change_point),
                    np.random.normal(loc=5, scale=2, size=size - change_point),
                )
            )

            gaussian_likelihood.learn(list(data[:learning_steps]))

            for time in range(learning_steps, size):
                observation = float(data[time])
                pred_probs = np.array(gaussian_likelihood.predict(observation))

                if time == time_after_learning:
                    self.assertTrue(
                        pred_probs[-1] > threshold_after_learning,
                        msg=f"Got too small ({pred_probs[-1]}) predictive probability after learning",
                    )

                if time == time_before_change_point:
                    self.assertTrue(
                        pred_probs[-1] >= threshold_change_point,
                        msg=f"Got too small ({pred_probs[-1]}) predictive probability before change point",
                    )

                if time == change_point:
                    self.assertTrue(
                        pred_probs[-1] < threshold_change_point,
                        msg=f"Got too large ({pred_probs[-1]}) predictive probability after change point",
                    )

                expected_size = time - learning_steps + 1
                self.assertTrue(
                    pred_probs.shape[0] == expected_size,
                    msg=f"Expected {expected_size} probabilities, got {pred_probs.shape[0]}",
                )

                gaussian_likelihood.update(observation)

    def test_clear(self):
        np.random.seed(1)

        gaussian_likelihood = GaussianUnknownMeanAndVariance()

        size = 51
        data = np.random.normal(loc=0, scale=1, size=size)

        gaussian_likelihood.learn(list(data[:-2]))
        first = gaussian_likelihood.predict(float(data[-1]))

        gaussian_likelihood.clear()

        gaussian_likelihood.learn(list(data[:-2]))
        second = gaussian_likelihood.predict(float(data[-1]))

        self.assertTrue(first == second, msg=f"Clear should keep results similar; {first} is not equal to {second}")


class TestSimpleDetector(unittest.TestCase):
    def test_detection(self):
        size = 100
        run_lengths = np.full((size,), 0.01)

        threshold = 0.04
        simple_detector = SimpleDetector(threshold)

        self.assertTrue(simple_detector.detect(run_lengths), msg="Here should be detected a change point")

    def test_clear(self):
        size = 100
        run_lengths = np.full((size,), 0.01)

        threshold = 0.04
        simple_detector_consecutive = SimpleDetector(threshold)
        simple_detector_clear = SimpleDetector(threshold)

        consecutive_result = simple_detector_consecutive.detect(run_lengths) and simple_detector_consecutive.detect(
            run_lengths
        )

        clear_result = simple_detector_clear.detect(run_lengths)
        simple_detector_clear.clear()
        clear_result = clear_result and simple_detector_clear.detect(run_lengths)

        self.assertTrue(
            consecutive_result == clear_result,
            msg=f"Clear should keep results similar; {consecutive_result} is not equal to {clear_result}",
        )


class TestSimpleLocalizer(unittest.TestCase):
    def test_localization(self):
        size = 11
        run_lengths = np.full((size,), 0.05)
        correct_run_length = 5
        run_lengths[correct_run_length] = 0.5

        simple_localizer = SimpleLocalizer()
        result = simple_localizer.localize(run_lengths)
        self.assertTrue(
            result == correct_run_length, msg=f"Run length should be {correct_run_length}, got {result} instead"
        )
