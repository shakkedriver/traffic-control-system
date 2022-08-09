from Agent import Agent
ACCELERATION_FACTOR = 1
import numpy as np
JUNCTION_SIZE = 50

class LightTrafficAgent(Agent):

    def __init__(self, env):
        super().__init__(env)
        self.cur_green_path = 0
        self.counter = 0

    def send_control_signal(self):
        # speed_state, age_state = self.env.get_state()
        len_without_junc = self.env.length - JUNCTION_SIZE - 5
        for car in self.env.cars_iterator():
            closest = min(self.get_closest_car_in_path(car), len_without_junc - car.dist)

            if car.path == self.cur_green_path:
                car.update_speed(ACCELERATION_FACTOR)
            elif car.speed ** 2 - 2 * closest >= -2:
                car.update_speed(-2 * ACCELERATION_FACTOR)
            else:
                car.update_speed(0)
        self.counter += 1
        if self.counter % 100 > 82:
            self.cur_green_path = self.cur_green_path + 1
        if self.counter % 100 == 0:
            self.cur_green_path = (self.cur_green_path + 1) % self.env.num_paths

    def get_closest_car_in_path(self, car):
        speed_state, age_state = self.env.get_state()
        path = speed_state[car.path][car.dist + 1:]
        cars = np.where(path != -1)[0]
        if cars.__len__() == 0:
            return self.env.length
        closest = cars[0]
        return closest - 3
