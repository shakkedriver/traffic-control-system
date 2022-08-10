from CarFactory import CarFactory
import random
import numpy as np

from NormalCar import NormalCar
from Car import CAR_MAX_SPEED
NORMAL_CAR_MAX_INIT_SPEED = CAR_MAX_SPEED
NORMAL_CAR_MIN_INIT_SPEED = 5


class NormalCarFactory(CarFactory):
    def __init__(self, path, creation_frequency):
        super().__init__(path)
        self.creation_frequency = creation_frequency

    def create_car(self, env):
        generate_car = random.random() < self.creation_frequency
        closest_car = self.get_closest_car_in_path(env)
        if closest_car / 1.5 <= NORMAL_CAR_MIN_INIT_SPEED:  # last car is too close
            return None
        if not generate_car:
            return None
        else:
            speed = random.randint(NORMAL_CAR_MIN_INIT_SPEED, int(min(NORMAL_CAR_MAX_INIT_SPEED, closest_car / 1.5)))
            car = NormalCar(self.path,0,speed,0)
            return car

    def get_closest_car_in_path(self, env):
        speed_state, age_state = env.get_state()
        path = speed_state[self.path]
        cars = np.where(path != -1)[0]
        if cars.__len__() == 0:
            return len(path)
        closest = cars[0]
        return closest

