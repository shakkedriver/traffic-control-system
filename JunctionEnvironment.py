from abc import ABC, abstractmethod

import numpy as np

import Car


class JunctionEnvironment(ABC):
    """
    this is an abstract class that represents our junction environment
    """

    def __init__(self, num_paths, length):

        self.num_paths = num_paths
        self.length = length
        # this is a list  of lists of all the cars in the environment each list represents a different paths
        self.cars = [dict() for i in range(num_paths)]
        score = 0

    @abstractmethod
    def propagate(self):
        """
        propagate all the cars in the environment create new car and checks for coalitions deals with them and update score
        :return: Report
        """
        pass

    @abstractmethod
    def check_collisions(self):
        """
        detect collisions
        :return: a list of tuples with the cars that collided
        """
        pass

    @abstractmethod
    def generate_new_cars(self):
        pass

    def get_state(self):
        """
        return the state of the enviorment
        """
        speed_state = np.full((self.num_paths, self.length), -1)
        age_state = np.ones(self.num_paths, self.length)
        for car in self.cars_iterator():
            speed_state[car.path][self.length - car.dist] = car.speed
            age_state[car.path][self.length - car.dist] = car.age
        return speed_state, age_state

    @abstractmethod
    def get_score_for_round(self, report):
        pass
        # shakked
    def delete_car(self,car):
        if id(car) not in self.cars[car.path].values():
            return
        self.cars[car.path].pop(id(car))

    def cars_iterator(self):
        for i in self.num_paths:
            cars = self.cars[i]
            for car in cars.values():
                yield car
