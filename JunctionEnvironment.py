from abc import ABC, abstractmethod

import numpy as np


class JunctionEnvironment(ABC):
    """
    this is an abstract class that represents our junction environment
    """

    def __init__(self, num_paths, length):

        self.num_paths = num_paths
        self.length = length
        # this is a list  of lists of all the cars in the environment each list represents a different paths
        self.cars = [[] for i in range(num_paths)]
        score = 0

    @abstractmethod
    def propergate(self):
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

    def get_state(self):
        #moshe
        speed_state = np.ones(self.num_paths, self.length)
        age_state = np.ones(self.num_paths, self.length)
        speed_state *= -1
        for i in self.num_paths:
            cars = self.cars[i]
            for car in cars:
                speed_state[car.path][self.length - car.dist] = car.speed
                age_state[car.path][self.length - car.dist] = car.age
        return speed_state, age_state
    
    def get_score_for_round(self,report):
        pass
        #shakked
