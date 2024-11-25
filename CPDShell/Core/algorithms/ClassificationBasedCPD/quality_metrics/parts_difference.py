"""
Module for implementation of classifier's quality metric based on difference between left and right parts.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable

import numpy as np

from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.iclassifier import Classifier
from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.iquality_metric import QualityMetric


class PartsDifference(QualityMetric):
    """
    The class implementing test statistic based on threshold overcome.
    """

    def __init__(self, classifier: Classifier) -> None:
        """
        Initializes a new instance of parts difference quality metric.

        :param classifier: Classifier that is going to be used for samples classification.
        """
        self.__classifier = classifier

    def assess_with_barrier(self, window: Iterable[float | np.float64], time: int) -> float:
        """Evaluates quality function based on classificator in the specified point.

        :param window: sample to be classified.
        :param time: Index of barrier in the given sample to calculate quality.
        :return: Quality assessment.
        """
        train_sample_X = [[x] for i, x in enumerate(window) if i % 2 == 0]
        train_sample_Y = [int(i > time) for i in range(0, len(window), 2)]
        test_sample = [[x] for i, x in enumerate(window) if i % 2 != 0]

        classified = self.__classifier.classify(train_sample_X, train_sample_Y, test_sample)
        right = classified.sum()
        left = len(classified) - right

        return 2 * min(right, left) / len(classified)
