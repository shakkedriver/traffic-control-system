from abc import ABC
from dataclasses import dataclass
import numpy as np


class Report(ABC):
    pass


@dataclass
class RegularReport(Report):
    passed_cars: list
    collisions_in_paths: list
    collisions_in_Junction: list
    late_cars: list
    speed_state: np.ndarray
    age_state: np.ndarray
    actions: list
