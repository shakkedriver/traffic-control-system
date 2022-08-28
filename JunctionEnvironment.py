from abc import ABC, abstractmethod
from itertools import combinations, product

import numpy as np

from CarFactory import NormalCarFactory
from Report import RegularReport


class JunctionEnvironment(ABC):
    """
    this is an abstract class that represents our junction environment
    """

    def __init__(self, num_paths, length, params):

        self.num_paths = num_paths
        self.length = length
        # this is a list  of lists of all the cars in the environment each list represents a different paths
        self.cars = [dict() for i in range(num_paths)]
        score = 0
        self.params = params

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
        return the state of the environment
        """
        speed_state = np.full((self.num_paths, self.length + 1), -1)
        age_state = np.full((self.num_paths, self.length + 1), -1)
        for car in self.cars_iterator():
            speed_state[car.path, car.dist] = car.speed
            age_state[car.path, car.dist] = car.age
        return speed_state, age_state

    @abstractmethod
    def get_score_for_round(self, report):
        pass
        # shakked

    def delete_car(self, car):
        if id(car) not in self.cars[car.path].keys():
            return
        self.cars[car.path].pop(id(car))

    def cars_iterator(self):
        for i in range(self.num_paths):
            cars = self.cars[i]
            for car in cars.values():
                yield car


class NormalEnvironment(JunctionEnvironment):

    def __init__(self, num_paths, length, params):
        super().__init__(num_paths, length, params)
        self.last_report = None

        reward_params = self.params['reward_params']
        self.path_collision_penalty = reward_params['PATH_COLLISION_PENALTY']
        self.junction_collision_penalty = reward_params['JUNCTION_COLLISION_PENALTY']
        self.late_penalty = reward_params['LATE_PENALTY']
        self.late_threshold = reward_params['LATE_THRESHOLD']
        self.reward_for_speed = reward_params['REWARD_FOR_SPEED']
        self.junction_size = reward_params['JUNCTION_SIZE']
        self.reward_for_passed_car = reward_params['REWARD_FOR_PASSED_CAR']

        freq_params = self.params['frequency_params']
        self.mu_of_car_creation = freq_params['MU_OF_CAR_CREATION']
        self.sigma_of_car_creation = freq_params['SIGMA_OF_CAR_CREATION']
        self.min_freq = freq_params['MIN_FREQ']
        self.max_freq = freq_params['MAX_FREQ']
        self.stop_new_cars_if_close = bool(freq_params['STOP_NEW_CARS_IF_CLOSE'])

        self.car_factories = [NormalCarFactory(path, self.get_creation_frequency()) for path in
                              range(num_paths)]

    def get_creation_frequency(self):
        x = np.random.normal(self.mu_of_car_creation, self.sigma_of_car_creation, 1)[0]
        # print(min(max(x, MIN_FREQ), MAX_FREQ))
        return min(max(x, self.min_freq), self.max_freq)

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
        cars_in_junction = [
            [car for car in path.values() if self.length - self.junction_size < car.dist]
            for path in self.cars]
        for path1, path2 in combinations(cars_in_junction, 2):
            result += list(product(path1, path2))
        return result

    def propagate(self, actions=None):
        """
        propagate all the cars in the environment create new car and checks for coalitions deals with them and update score
        :return: Report
        """
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
        if actions is None:
            actions = []
        self.last_report = RegularReport(list_of_passed_cars, collisions_in_paths, collisions_in_junction,
                             late_cars, speed_state, age_state, actions)
        return self.last_report

    def get_last_report(self, actions):
        if self.last_report is None:
            return None
        self.last_report.actions = actions
        return self.last_report

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
            if car.age > self.late_threshold:
                late_cars.append(car)
            car.age += 1
        return late_cars

    def get_score_for_round(self, report: RegularReport):
        # total_time = sum(car.age for car in self.cars_iterator())
        total_speed = sum(car.speed for car in self.cars_iterator())
        return - len(report.collisions_in_paths) * self.path_collision_penalty \
               - len(report.collisions_in_Junction) * self.junction_collision_penalty \
               + total_speed * self.reward_for_speed \
               - len(report.late_cars) * self.late_penalty \
               + len(report.passed_cars) * self.reward_for_passed_car
