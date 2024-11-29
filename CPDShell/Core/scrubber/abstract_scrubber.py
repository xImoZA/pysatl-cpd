from abc import ABC, abstractmethod
from typing import Sequence, Iterable

import numpy

from CPDShell.Core.data_controller import DataController
from CPDShell.Core.scenario import Scenario


class AbstractScrubber(ABC):
    """A scrubber for dividing data into windows
    and subsequent processing of data windows
    by change point detection algorithms
    """

    def __init__(
            self,
            scenario: Scenario,
    ) -> None:
        """A scrubber for dividing data into windows
        and subsequent processing of data windows
        by change point detection algorithms

        :param scenario: :class:`Scenario` object with information about the scrubber task
        :param data_controller: data controller for getting values for change point detection
        """
        self.scenario = scenario
        self.data = []
        self.is_running = True
        self.change_points: list[int] = []

    @abstractmethod
    def restart(self) -> None:
        """Function for restarting Scrubber"""
        ...

    @abstractmethod
    def get_windows(self) -> Iterable[Sequence[float | numpy.float64]]:
        """Function for dividing data into parts to feed into the change point detection algorithm

        :return: Iterator of data windows for change point detection algorithm
        """
        ...

    @abstractmethod
    def add_change_points(self, window_change_points: list[int]) -> None:
        """Function for mapping window change points to scrubber data part"""
        ...

    def add_data(self, new_data: Sequence[float | numpy.float64]) -> None:
        """Function for adding new data to Scrubber"""
        self.data += new_data  # TODO Sequence __add__?
