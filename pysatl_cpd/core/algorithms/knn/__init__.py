"""
Module for knn CPD algorithm's customization blocks.
"""

__author__ = "Loikov Vladislav"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.core.algorithms.knn.abstracts import IObservation, Neighbour
from pysatl_cpd.core.algorithms.knn.classifier import KNNClassifier
from pysatl_cpd.core.algorithms.knn.graph import KNNGraph
from pysatl_cpd.core.algorithms.knn.heap import NNHeap

__all__ = [
    "IObservation",
    "KNNClassifier",
    "KNNGraph",
    "NNHeap",
    "Neighbour",
]
