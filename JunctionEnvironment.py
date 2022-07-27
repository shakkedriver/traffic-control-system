from abc import ABC, abstractmethod


class JunctionEnvironment(ABC):
    """
    this is an abstract class that represents our junction environment
    """

    def __init__(self, num_paths, length):

        self.num_paths = num_paths
        self.length = length
        # this is a list  of lists of all the cars in the environment each list represents a different paths
        self.cars = [[] for i in range(num_paths)]

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
