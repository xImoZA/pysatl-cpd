"""
Module for implementations of Bayesian CPD algorithm localizers.
"""

__author__ = "Alexey Tatyanenko, Loikov Vladislav"
__copyright__ = "Copyright (c) 2024 Alexey Tatyanenko"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.core.algorithms.bayesian.localizers.argmax import (
    ArgmaxLocalizer,
)

__all__ = [
    "ArgmaxLocalizer",
]
