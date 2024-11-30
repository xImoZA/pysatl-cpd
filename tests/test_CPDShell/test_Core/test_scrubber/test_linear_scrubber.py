import pytest

from CPDShell.Core.scenario import Scenario
from CPDShell.Core.scrubber.linear_scrubber import LinearScrubber


class TestLinearScrubber:
    @pytest.mark.parametrize(
        "scenario_param,data,window_length,expected_windows",
        (
            (
                (1, True),
                (1, 2, 3, 4, 5, 6, 7),
                5,
                [(1, 2, 3, 4, 5), (2, 3, 4, 5, 6), (3, 4, 5, 6, 7)],
            ),
            (
                (1, True),
                (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
                7,
                [(1, 2, 3, 4, 5, 6, 7), (3, 4, 5, 6, 7, 8, 9), (5, 6, 7, 8, 9, 10, 11), (7, 8, 9, 10, 11, 12, 13)],
            ),
        ),
    )
    def test_generate_window(self, scenario_param, data, window_length, expected_windows):
        scenario = Scenario(*scenario_param)
        scrubber = LinearScrubber(window_length)
        scrubber.scenario = scenario
        scrubber._data = data
        for window in scrubber.get_windows():
            assert window == expected_windows.pop(0)

    @pytest.mark.parametrize(
        "scenario_param,data,change_points,expected_change_points",
        (
            (
                (1, True),
                (1, 2, 3, 4, 5, 6, 7),
                [
                    [1, 2],
                ],
                [1],
            ),
            (
                (2, True),
                (1, 2, 3, 4, 5, 6, 7),
                [
                    [1, 2],
                ],
                [1, 2],
            ),
        ),
    )
    def test_add_change_points(self, scenario_param, data, change_points, expected_change_points):
        scenario = Scenario(*scenario_param)
        scrubber = LinearScrubber(100)
        scrubber.scenario = scenario

        scrubber._data = data
        for window in scrubber.get_windows():
            scrubber.add_change_points(change_points.pop(0))
        assert scrubber.change_points == expected_change_points
