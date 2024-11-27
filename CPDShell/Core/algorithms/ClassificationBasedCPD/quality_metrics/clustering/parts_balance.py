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
    The class implementing quality metric based on difference between left and right parts.
    """

    def assess_with_barrier(self, classifier: Classifier, sample: Iterable[float | np.float64], time: int) -> float:
        """Evaluates quality function based on classificator in the specified point.

        :param window: sample to be classified.
        :param time: Index of barrier in the given sample to calculate quality.
        :return: Quality assessment.
        """
        classifier.train(sample)
        predicted_classes = classifier.predict(list(sample))
        sample_length = len(predicted_classes)

        left_rate = sum(predicted_classes[0:time]) / time
        right_rate = sum(predicted_classes[time:sample_length]) / (sample_length - time)

        return abs(left_rate - right_rate)
