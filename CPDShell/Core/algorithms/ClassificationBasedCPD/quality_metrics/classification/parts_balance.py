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
        train_sample = [[x] for i, x in enumerate(sample) if i % 2 == 0]
        test_sample = [[x] for i, x in enumerate(sample) if i % 2 != 0]

        classifier.train(train_sample, time / 2)
        predicted_classes = classifier.predict(test_sample)
        sample_length = len(predicted_classes)
        
        right = predicted_classes.sum()
        left = sample_length - right

        return 2 * min(right, left) / sample_length
