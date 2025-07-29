"""
Module for implementations of knn CPD algorithm abstracts functions.
"""

__author__ = "Loikov Vladislav"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.core.algorithms.knn.abstracts.iobservation import (
    IObservation,
    Neighbour,
)

__all__ = [
    "IObservation",
    "Neighbour",
]
