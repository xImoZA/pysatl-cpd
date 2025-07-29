"""
Module for abstract base classes for classification CPD algorithm.
"""

__author__ = "Loikov Vladislav"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.core.algorithms.classification.abstracts.iclassifier import IClassifier
from pysatl_cpd.core.algorithms.classification.abstracts.iquality_metric import IQualityMetric
from pysatl_cpd.core.algorithms.classification.abstracts.istatistic_test import ITestStatistic

__all__ = [
    "IClassifier",
    "IQualityMetric",
    "ITestStatistic",
]
