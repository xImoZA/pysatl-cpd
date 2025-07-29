"""
Module contains protocol for change point detection algorithms' interface.
"""

__author__ = "Romanyuk Artem, Vladimir Kutuev"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Protocol

import numpy as np
import numpy.typing as npt


class Algorithm(Protocol):
    """Protocol for change point detection algorithms' interface"""

    def detect(self, window: npt.NDArray[np.float64]) -> int:
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: the number of change points in the window
        """
        ...

    def localize(self, window: npt.NDArray[np.float64]) -> list[int]:
        """Function for finding coordinates of change points in window

        :param window: part of global data for finding change points
        :return: list of window change points
        """
        ...
