from dataclasses import dataclass
from abc import ABC, abstractmethod

CAR_MAX_SPEED = 12
# @dataclass
class Car(ABC):
    """
    this class represents a car in out system. the car have a path which is the path it is on. a speed wich is the speed
     of the car. and dist which is the distance of the car from the junction.
    """
    # path: int
    # dist: int
    # speed: int
    # age: int
    # max_speed: int = 15

    def __init__(self, path, dist, speed, age, max_speed=CAR_MAX_SPEED):
        self.path = path
        self.dist = dist
        self.speed = speed
        self.age = age
        self.max_speed = max_speed

    @abstractmethod
    def update_speed(self, acceleration):
        pass