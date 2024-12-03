"""
Module for Data Controller testing.
"""

__author__ = "Romanyuk Artem"
__copyright__ = "Copyright (c) 2024 Romanyuk Artem"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from CPDShell.Core.data_controller import DataController


class TestDataController:
    @pytest.mark.parametrize(
        "data,window_length,change_points,data_start_index",
        (
            ((1, 2, 3, 4, 5, 6, 7), 5, [1, 2], 1),
            ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 10, [7], 10),
        ),
    )
    def test_restart(self, data, window_length, change_points, data_start_index):
        data_controller = DataController(data, window_length)
        data_controller.change_points = change_points
        data_controller._data_start_index = data_start_index
        data_controller.restart()
        assert data_controller._data_start_index == 0
        assert data_controller.change_points == []

    @pytest.mark.parametrize(
        "data,window_length,data_start_index,expected_window",
        (
            (
                (1, 2, 3, 4, 5, 6, 7),
                5,
                1,
                (2, 3, 4, 5, 6),
            ),
            (
                (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
                10,
                2,
                (3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
            ),
        ),
    )
    def test_get_data(self, data, window_length, data_start_index, expected_window):
        data_controller = DataController(data, window_length)
        data_controller._data_start_index = data_start_index
        assert next(iter(data_controller.get_data())) == expected_window

    @pytest.mark.parametrize(
        "data,window_length,window_start_index,change_points,expected_change_points",
        (
            ((1, 2, 3, 4, 5, 6, 7), 5, 1, [2], [3]),
            ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 10, 2, [4, 8], [6, 10]),
        ),
    )
    def test_add_change_points(self, data, window_length, window_start_index, change_points, expected_change_points):
        data_controller = DataController(data, window_length)
        data_controller._data_start_index = window_start_index
        data_controller.add_change_points(change_points)
        assert data_controller.change_points == expected_change_points
