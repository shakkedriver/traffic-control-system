from Agent import Agent
import numpy as np
from NormalEnvironment import JUNCTION_SIZE
from NormalCar import DECELERATION_FACTOR
LEN_OF_GREEN = 100
DELAY_BETWEEN_GREENS = 15


class LightTrafficAgent(Agent):

    def __init__(self, env):
        super().__init__(env)
        self.cur_green_path = 0
        self.counter = 0
        self.all_red = False
        self.len_without_junc = self.env.length - JUNCTION_SIZE - 1

    def send_control_signal(self):
        # speed_state, age_state = self.env.get_state()
        for car in self.env.cars_iterator():
            if (car.path == self.cur_green_path and not self.all_red) or car.path >= self.len_without_junc:
                closest = self.get_closest_car_in_path(car)
            else:
                closest = min(self.get_closest_car_in_path(car), self.len_without_junc - car.dist)
            if car.speed ** 2 + DECELERATION_FACTOR * closest >= -3:
                car.update_speed(-1)
            # elif(closest):
            else:
                car.update_speed(1)
        self.counter += 1
        if (self.counter % LEN_OF_GREEN) > LEN_OF_GREEN - DELAY_BETWEEN_GREENS:
            self.all_red = True
        if (self.counter % LEN_OF_GREEN) == 0:
            self.all_red = False
            self.cur_green_path = (self.cur_green_path + 1) % self.env.num_paths

    def get_closest_car_in_path(self, car):
        speed_state, age_state = self.env.get_state()
        path = speed_state[car.path][car.dist + 1:]
        cars = np.where(path != -1)[0]
        if cars.__len__() == 0:
            return self.env.length * 2
        closest = cars[0]
        return closest - 3
