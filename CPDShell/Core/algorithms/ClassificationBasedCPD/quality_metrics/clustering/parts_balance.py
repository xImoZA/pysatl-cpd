"""
Module for implementation of clustering quality metric based on balance between left and right parts.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np

from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.iquality_metric import QualityMetric


class PartsBalance(QualityMetric):
    """
    The class implementing quality metric based on balance between left and right parts.
    """

    def assess_barrier(self, classes: list[float | np.float64], time: int) -> float:
        """Evaluates quality function based on classificator in the specified point.

        :param classes: Classes of observations, predicted by the classifier.
        :param time: Index of barrier in the given sample to calculate quality.
        :return: Quality assessment.
        """
        sample_length = len(classes)

        left_rate = sum(classes[0:time]) / time
        right_rate = sum(classes[time:sample_length]) / (sample_length - time)

        return abs(left_rate - right_rate)
