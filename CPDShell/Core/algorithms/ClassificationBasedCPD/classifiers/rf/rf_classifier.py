"""
Module for implementation of random forest classifier for cpd.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.iclassifier import Classifier


class RFClassifier(Classifier):
    """
    The class implementing random forest classifier for cpd.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initializes a new instance of k-means classifier for cpd.
        """
        self.__model: RandomForestClassifier | None = None

    def classify(
        self,
        train_X: list[float | np.float64],
        train_Y: list[float | np.float64],
        to_classify: Iterable[float | np.float64],
    ) -> np.ndarray:
        """Applies classificator to the given sample.

        :param window: part of global data for finding change points.
        """
        self.__model = RandomForestClassifier()
        self.__model.fit(train_X, train_Y)
        prediction = self.__model.predict(to_classify)

        return prediction
