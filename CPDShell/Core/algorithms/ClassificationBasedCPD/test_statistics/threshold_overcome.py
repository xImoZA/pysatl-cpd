"""
Module for implementation of CPD algorithm based on nearest neighbours.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import typing as tp
from collections import deque
from collections.abc import Iterable
from math import sqrt

import numpy as np

import CPDShell.Core.algorithms.KNNCPD.knn_graph as knngraph
from CPDShell.Core.algorithms.abstract_algorithm import Algorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.istatistic_test import StatisticTest


class ThresholdOvercome(StatisticTest):
    def __init__(
        self,
        threshold: float
    ) -> None:
        """
        Initializes a new instance of KNN change point algorithm.

        :param metric: function for calculating distance between points in time series.
        :param k: number of neighbours in graph relative to each point.
        """
        self.__threshold = threshold

    def get_change_points(self, classifier_assesments: list[float]) -> list[int]:
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: the number of change points in the window
        """
        return [i for i, v in enumerate(classifier_assesments) if v > self.__threshold]
        # return list(filter(lambda x : x > self.__threshold, classifier_assesments))
