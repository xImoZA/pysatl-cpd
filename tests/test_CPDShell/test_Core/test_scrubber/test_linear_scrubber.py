import random

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from CPDShell.Core.scrubber.linear_scrubber import LinearScrubber
from CPDShell.Core.scrubberscenario import ScrubberScenario


class TestLinearScrubber:
    @settings(max_examples=1000)
    @given(st.integers(0, 100), st.integers(0, 100), st.floats(0.01, 1))
    def test_generate_window(self, data_length, window_length, shift_factor):
        data = [i for i in range(data_length)]
        scenario = ScrubberScenario(1, True)
        scrubber = LinearScrubber(window_length, shift_factor)
        scrubber.scenario = scenario
        scrubber._data = data
        cur_index = 0
        for window in scrubber.get_windows():
            assert window == data[cur_index : cur_index + window_length]
            cur_index += max(1, int(window_length * shift_factor))

    @settings(max_examples=1000)
    @given(st.integers(0, 100), st.integers(0, 100), st.integers(0, 100), st.floats(0.01, 1))
    def test_add_change_points(self, max_change_point_number, data_length, window_length, shift_factor):
        data = [i for i in range(data_length)]
        scenario = ScrubberScenario(max_change_point_number, True)
        scrubber = LinearScrubber(window_length, shift_factor)
        scrubber.scenario = scenario
        scrubber._data = data

        cur_index = 0
        for window in scrubber.get_windows():
            change_points_num = random.randint(0, window_length)
            change_points = [random.randint(0, window_length - 1) for _ in range(change_points_num)]
            scrubber.add_change_points(change_points)
            for point in change_points[:max_change_point_number]:
                assert point + cur_index in scrubber.change_points
            if max_change_point_number <= change_points_num:
                assert not scrubber.is_running
            cur_index += max(1, int(window_length * shift_factor))

    def test_scrubber_scenario_exception(self):
        scrubber = LinearScrubber()
        with pytest.raises(ValueError):
            scrubber.add_change_points([])
