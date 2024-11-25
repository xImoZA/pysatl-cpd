"""
Module for Classification CPD algorithm's classifier abstract base class.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np


class Classifier(ABC):
    """Classifier's abstract base class."""

    @abstractmethod
    def classify(
        self,
        train_X: list[float | np.float64],
        train_Y: list[float | np.float64],
        to_classify: Iterable[float | np.float64],
    ) -> np.ndarray:
        """Applies classificator to the given sample.

        :param window: part of global data for finding change points.
        """
        raise NotImplementedError
