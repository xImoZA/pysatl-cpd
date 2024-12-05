from dataclasses import dataclass


@dataclass
class ScrubberScenario:
    """Scenario for scrubber

    :param max_change_point_number: maximum number of change points on the scrubber window, defaults to 1
    :param to_localize: is it necessary to localize change points, defaults to False
    """

    max_change_point_number: int = 1
    to_localize: bool = False
