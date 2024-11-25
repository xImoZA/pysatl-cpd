"""
Module for implementation of svm classifier for cpd.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable

import numpy as np
from sklearn.svm import SVC

from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.iclassifier import Classifier


class SVMAlgorithm(Classifier):
    """
    The class implementing svm classifier for cpd.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initializes a new instance of k-means classifier for cpd.
        """
        self.__model: SVC | None = None

    def classify(
        self,
        train_X: list[float | np.float64],
        train_Y: list[float | np.float64],
        to_classify: Iterable[float | np.float64],
    ) -> np.ndarray:
        """Applies classificator to the given sample.

        :param window: part of global data for finding change points.
        """
        self.__model = SVC()
        self.__model.fit(train_X, train_Y)
        prediction = self.__model.predict(to_classify)

        return prediction
