from JunctionEnvironment import JunctionEnvironment
from NormalCarFactory import NormalCarFactory

JUNCTION_SIZE = 50


class NormalEnvironment(JunctionEnvironment, ):
    def __init__(self, num_paths, length):
        super().__init__(num_paths, length)
        self.car_factories = [NormalCarFactory(path) for path in range(num_paths)]

    def check_collisions(self):
        collisions = self.__check_collisions_in_paths()
        collisions += self.__check_collisions_in_Junction()
        return collisions

    def __check_collisions_in_paths(self):
        collisions = []
        for path in self.cars:
            for first, second in zip(path, path[1:]):
                if first.dist >= second.dist:
                    collisions.append((first, second))

    def __check_collisions_in_Junction(self):
        cars_in_junction = set()
        for path in self.cars:
            for car in path:
                if car.dist>self.length-JUNCTION_SIZE:
                    cars_in_junction.add(car)

