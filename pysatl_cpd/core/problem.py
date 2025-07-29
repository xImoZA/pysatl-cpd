"""
Module contains problem specification for change point detection tasks.
"""

__author__ = "Vladimir Kutuev, Loikov Vladislav"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"
from dataclasses import dataclass


@dataclass
class CpdProblem:
    """Specification of the solving problem

    :param to_localize: is it necessary to localize change points, defaults to False
    """

    to_localize: bool = True
