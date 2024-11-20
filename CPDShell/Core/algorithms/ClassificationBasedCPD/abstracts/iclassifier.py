from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np


class Classifier(ABC):
    """Abstract class for change point detection algorithms"""

    @abstractmethod
    def classify(self, window: Iterable[float | np.float64]) -> None:
        """Applies classificator to the given sample.

        :param window: part of global data for finding change points
        """
        raise NotImplementedError

    @abstractmethod
    def assess_barrier(self, time: int) -> float:
        """Evaluates quality function based on classificator in the specified point.

        :param window: Index of point in the given sample to calculate quality.
        :return: Quality assessment.
        """
        raise NotImplementedError
