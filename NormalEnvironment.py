from JunctionEnvironment import JunctionEnvironment
from NormalCarFactory import NormalCarFactory
from itertools import combinations, product
from RegularReport import RegularReport
import numpy as np

# reward:
PATH_COLLISION_PENALTY = 10000
JUNCTION_COLLISION_PENALTY = 10000
LATE_PENALTY = 25
LATE_THRESHOLD = 100  # should be some fraction of the length of the environment #todo
REWARD_FOR_SPEED = 1000 / 150
JUNCTION_SIZE = 6
REWARD_FOR_PASSED_CAR = 1000
# frequency parameters:
MU_OF_CAR_CREATION = 0.08
SIGMA_OF_CAR_CREATION = 0.04
MIN_FREQ = 0.02
MAX_FREQ = 0.3
LENGTH_OF_PATH = 50

class NormalEnvironment(JunctionEnvironment):
    def __init__(self, num_paths, length=LENGTH_OF_PATH):
        super().__init__(num_paths, length)
        self.car_factories = [NormalCarFactory(path, self.get_creation_frequency()) for path in range(num_paths)]

    def get_creation_frequency(self):
        x = np.random.normal(MU_OF_CAR_CREATION, SIGMA_OF_CAR_CREATION, 1)[0]
        # print(min(max(x, MIN_FREQ), MAX_FREQ))
        return min(max(x, MIN_FREQ), MAX_FREQ)

    def check_collisions(self):
        collisions_in_paths = self.__check_collisions_in_paths()
        collisions_in_junction = self.__check_collisions_in_Junction()
        return collisions_in_paths, collisions_in_junction

    def __check_collisions_in_paths(self):
        """return a list of all the tuples of cars that collided on the same path ie one car pased another car"""
        collisions = []
        for path in self.cars:
            for first, second in zip(list(path.values()), list(path.values())[1:]):
                if first.dist <= second.dist:
                    collisions.append((first, second))
        return collisions

    def __check_collisions_in_Junction(self):
        """return a list of all the tuples of cars that collided in the junction ie one car was at the junction while
        another was in a different lane """
        result = []
        cars_in_junction = [[car for car in path.values() if self.length - JUNCTION_SIZE < car.dist]
                            for path in self.cars]
        for path1, path2 in combinations(cars_in_junction, 2):
            result += list(product(path1, path2))
        return result

    def propagate(self, actions):
        """
        propagate all the cars in the environment create new car and checks for coalitions deals with them and update score
        :return: Report
        """
        # shakked

        list_of_passed_cars = self.__move_cars()
        for car in list_of_passed_cars:
            self.delete_car(car)
        collisions_in_paths, collisions_in_junction = self.check_collisions()
        for first, second in collisions_in_paths:
            self.delete_car(first)
            self.delete_car(second)
        for first, second in collisions_in_junction:
            self.delete_car(first)
            self.delete_car(second)
        self.generate_new_cars()
        late_cars = self.__get_late_cars_and_increment_age()
        speed_state, age_state = self.get_state()
        return RegularReport(list_of_passed_cars, collisions_in_paths, collisions_in_junction,
                             late_cars, speed_state, age_state, actions)

    def __move_cars(self):
        """
        move all the cars according to their speeds cars that are out of the environment are recorded and returned as a list
        """
        list_of_passed_cars = []
        for car in self.cars_iterator():
            car.dist += car.speed
            if car.dist >= self.length:
                list_of_passed_cars.append(car)
        return list_of_passed_cars

    def generate_new_cars(self):
        """
        create new cars in the environment
        :return:
        """
        for i, path in enumerate(self.cars):
            new_car = self.car_factories[i].create_car(self)
            if not new_car:
                continue
            path[id(new_car)] = new_car

    def __get_late_cars_and_increment_age(self):
        """return a list of all the late cars in the environment"""
        late_cars = []
        for car in self.cars_iterator():
            if car.age > LATE_THRESHOLD:
                late_cars.append(car)
            car.age += 1
        return late_cars

    def get_score_for_round(self, report: RegularReport):
        # total_time = sum(car.age for car in self.cars_iterator())
        total_speed = sum(car.speed for car in self.cars_iterator())
        return -len(report.collisions_in_paths) * PATH_COLLISION_PENALTY - len(
            report.collisions_in_Junction) * JUNCTION_COLLISION_PENALTY + total_speed * REWARD_FOR_SPEED - len(
            report.late_cars) * LATE_PENALTY + len(report.passed_cars) * REWARD_FOR_PASSED_CAR
