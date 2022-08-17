from dataclasses import dataclass

from Report import Report


@dataclass
class RegularReport(Report):
    passed_cars: list
    collisions_in_paths: list
    collisions_in_Junction: list
    late_cars: list
    speed_state: list
    age_state: list
    actions: dict
