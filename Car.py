from dataclasses import dataclass
from abc import ABC, abstractmethod

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

    def __init__(self, path, dist, speed, age, max_speed=15):
        self.path = path
        self.dist = dist
        self.speed = speed
        self.age = age
        self.max_speed = max_speed

    @abstractmethod
    def update_speed(self, acceleration):
        pass