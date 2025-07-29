"""
Module for core CPD algorithm's customization blocks.
"""

__author__ = "Loikov Vladislav"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.core.cpd_core import CpdCore
from pysatl_cpd.core.online_cpd_core import OnlineCpdCore
from pysatl_cpd.core.problem import CpdProblem

__all__ = [
    "CpdCore",
    "CpdProblem",
    "OnlineCpdCore",
]
