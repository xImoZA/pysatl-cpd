"""
Module for algorithms CPD algorithm's customization blocks.
"""

__author__ = "Loikov Vladislav"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.core.algorithms.abstract_algorithm import Algorithm
from pysatl_cpd.core.algorithms.bayesian_algorithm import BayesianAlgorithm
from pysatl_cpd.core.algorithms.bayesian_linear_heuristic import BayesianLinearHeuristic
from pysatl_cpd.core.algorithms.bayesian_online_algorithm import BayesianOnline
from pysatl_cpd.core.algorithms.classification_algorithm import ClassificationAlgorithm
from pysatl_cpd.core.algorithms.graph_algorithm import GraphAlgorithm
from pysatl_cpd.core.algorithms.kliep_algorithm import KliepAlgorithm
from pysatl_cpd.core.algorithms.knn_algorithm import KNNAlgorithm
from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm
from pysatl_cpd.core.algorithms.rulsif_algorithm import RulsifAlgorithm

__all__ = [
    "Algorithm",
    "BayesianAlgorithm",
    "BayesianLinearHeuristic",
    "BayesianOnline",
    "ClassificationAlgorithm",
    "GraphAlgorithm",
    "KNNAlgorithm",
    "KliepAlgorithm",
    "OnlineAlgorithm",
    "RulsifAlgorithm",
]
