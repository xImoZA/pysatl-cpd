"""
Module for Classification CPD algorithm's quality metric abstract base class.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np

from .iclassifier import Classifier


class QualityMetric(ABC):
    """Quality metric's abstract base class."""

    @abstractmethod
    def __init__(self, classifier: Classifier) -> None:
        """
        Initializes a new instance of parts difference quality metric.

        :param classifier: Classifier that is going to be used for samples classification.
        """
        self.__classifier = classifier

    @abstractmethod
    def assess_with_barrier(self, window: Iterable[float | np.float64], time: int) -> float:
        """Evaluates quality function based on classificator in the specified point.

        :param window: sample to be classified.
        :param time: Index of barrier in the given sample to calculate quality.
        :return: Quality assessment.
        """
        raise NotImplementedError
