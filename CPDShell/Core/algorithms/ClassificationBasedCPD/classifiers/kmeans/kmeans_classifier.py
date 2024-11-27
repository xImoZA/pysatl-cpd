"""
Module for implementation of kmeans classifier for cpd.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable

import numpy as np
from sklearn.cluster import KMeans

from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.iclassifier import Classifier


class KMeansAlgorithm(Classifier):
    """
    The class implementing k-means classifier for cpd.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initializes a new instance of k-means classifier for cpd.
        """
        self.__model: KMeans | None = None
        self.__sample: list[float | np.float64] | None = None

    def train(
        self,
        sample: list[float | np.float64],
        barrier: int
    ) -> None:
        """Trains classifier on the given sample.

        :param sample: sample for training classifier.
        """
        if self.__sample == sample:
            return

        k_means = KMeans(n_clusters=2)
        self.__sample = sample
        window_reshaped = np.array(sample).reshape(-1, 1)
        self.__model = k_means.fit(window_reshaped)

    def predict(
        self,
        sample: list[float | np.float64]
    ) -> np.ndarray:
        """Applies classificator to the given sample.

        :param window: part of global data for finding change points.
        """
        return self.__model.labels_
