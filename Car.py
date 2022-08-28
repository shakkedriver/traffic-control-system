from abc import ABC, abstractmethod

CAR_MAX_SPEED = 12


class Car(ABC):
    """
    this class represents a car in out system. the car have a path which is the path it is on. a speed
     which is the speed of the car. and dist which is the distance of the car from the junction.
    """

    def __init__(self, path, dist, speed, age, max_speed=CAR_MAX_SPEED):
        self.path = path
        self.dist = dist
        self.speed = speed
        self.age = age
        self.max_speed = max_speed

    @abstractmethod
    def update_speed(self, acceleration):
        pass


class NormalCar(Car):
    ACCELERATION_FACTOR = 1
    DECELERATION_FACTOR = -2

    def update_speed(self, acceleration):
        if acceleration > 0:
            self.speed += self.ACCELERATION_FACTOR
        elif acceleration < 0:
            self.speed += self.DECELERATION_FACTOR
        self.speed = max(0, min(self.max_speed, self.speed))
