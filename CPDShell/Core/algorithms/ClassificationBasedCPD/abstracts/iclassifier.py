from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy


class Classifier(ABC):
    """Abstract class for change point detection algorithms"""

    @abstractmethod
    def classify(self, window: Iterable[float | numpy.float64]) -> None:
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: the number of change points in the window
        """
        raise NotImplementedError

    @abstractmethod
    def quantify_in_point(self, time: int) -> float:
        """Function for finding coordinates of change points in window

        :param window: part of global data for finding change points
        :return: list of window change points
        """
        raise NotImplementedError
