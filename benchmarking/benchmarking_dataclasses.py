from dataclasses import dataclass


@dataclass
class OnlineCpdBenchmarkingResult:
    change_point: int
    delay: int
