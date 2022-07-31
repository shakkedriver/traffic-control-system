from JunctionEnvironment import JunctionEnvironment
from NormalCarFactory import NormalCarFactory
from itertools import combinations, product

JUNCTION_SIZE = 50



class NormalEnvironment(JunctionEnvironment):
    def __init__(self, num_paths, length):
        super().__init__(num_paths, length)
        self.car_factories = [NormalCarFactory(path) for path in range(num_paths)]

    def check_collisions(self):
        collisions = self.__check_collisions_in_paths()
        collisions += self.__check_collisions_in_Junction()
        return collisions

    def __check_collisions_in_paths(self):
        """return a list of all the tuples of cars that collided on the same path ie one car pased another car"""
        collisions = []
        for path in self.cars:
            for first, second in zip(path, path[1:]):
                if first.dist >= second.dist:
                    collisions.append((first, second))

    def __check_collisions_in_Junction(self):
        """return a list of all the tuples of cars that collided in the junction ie one car was at the junction while
        another was in a different lane """
        result = []
        cars_in_junction = [[car for car in path if car.dist > self.length - JUNCTION_SIZE] for path in self.cars]
        for path1, path2 in combinations(cars_in_junction, 2):
            result += list(product(path1, path2))
        return result



    def propergate(self):
        """
        propagate all the cars in the environment create new car and checks for coalitions deals with them and update score
        :return: Report
        """
        #shakked
        report = RegularReport()
        self.__move_cars()

    def __move_cars(self):
        pass

