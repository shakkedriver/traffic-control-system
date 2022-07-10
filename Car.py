from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Car(ABC):
    """
    this class represents a car in out system. the car have a path which is the path it is on. a speed wich is the speed
     of the car. and dist which is the distance of the car from the junction.
    """
    path: int
    dist: int
    speed: int
    @abstractmethod
    def update_speed(self, speed):
        pass