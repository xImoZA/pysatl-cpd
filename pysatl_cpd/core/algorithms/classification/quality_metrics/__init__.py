"""
Module for qualiti_metrics CPD algorithm's customization blocks.
"""

__author__ = "Loikov Vladislav"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.core.algorithms.classification.quality_metrics.classification import (
    F1,
    MCC,
    Accuracy,
)

__all__ = [
    "F1",
    "MCC",
    "Accuracy",
]
