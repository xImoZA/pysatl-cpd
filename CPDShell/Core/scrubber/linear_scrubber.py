from typing import Sequence, Iterable

import numpy

from CPDShell.Core.scenario import Scenario
from CPDShell.Core.scrubber.abstract_scrubber import AbstractScrubber


class LinearScrubber(AbstractScrubber):
    """A linear scrubber for dividing data into windows by moving them through data"""

    def __init__(self, scenario: Scenario,
                 window_length: int = 100,
                 movement_k: float = 1.0 / 3.0, ):
        """A linear scrubber for dividing data into windows by moving them through data

        :param scenario: :class:`Scenario` object with information about the scrubber task
        :param window_length: length of data window
        :param movement_k: how far will the window move relative to the length
        """
        super().__init__(scenario)
        self._window_length = window_length
        self._movement_k = movement_k
        self._window_start = 0

    def restart(self) -> None:
        self.change_points = []
        self.is_running = True

    def get_windows(self) -> Iterable[Sequence[float | numpy.float64]]:
        if self.data:
            window_end = self._window_start + self._window_length
            yield self.data[self._window_start:window_end]
            self._window_start += int(self._window_length * self._movement_k)
        while self._window_start + self._window_length <= len(self.data) and self.is_running:
            window_end = self._window_start + self._window_length
            yield self.data[self._window_start:window_end]
            self._window_start += int(self._window_length * self._movement_k)

    def add_change_points(self, window_change_points: list[int]) -> None:
        max_change_points = self.scenario.change_point_number
        change_point_number = max(0, max_change_points - len(self.change_points))
        if change_point_number == 0:
            self.is_running = False
        self.change_points += list(map(lambda point: self._window_start + point,
                                       window_change_points[:change_point_number]))