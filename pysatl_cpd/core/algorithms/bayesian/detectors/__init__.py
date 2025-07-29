"""
Module for implementations of Bayesian CPD algorithm detectors.
"""

__author__ = "Alexey Tatyanenko, Loikov Vladislav"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.core.algorithms.bayesian.detectors.drop import DropDetector
from pysatl_cpd.core.algorithms.bayesian.detectors.threshold import ThresholdDetector

__all__ = [
    "DropDetector",
    "ThresholdDetector",
]
