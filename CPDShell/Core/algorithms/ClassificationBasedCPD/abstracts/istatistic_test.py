from abc import ABC, abstractmethod


class StatisticTest(ABC):
    """Abstract class for change point detection algorithms"""

    @abstractmethod
    def get_change_points(self, classifier_assesments: list[float]) -> list[int]:
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: the number of change points in the window
        """
        raise NotImplementedError
