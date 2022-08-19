from Agent import Agent
import numpy as np
from NormalEnvironment import JUNCTION_SIZE
from NormalCar import DECELERATION_FACTOR
# LEN_OF_GREEN = 100
# DELAY_BETWEEN_GREENS = 15
from featureExtractors import SimpleExtractor
from qlearningAgents import ApproximateQAgent

POSSIBLE_ACTIONS = [[-1, -1],
                    [-1, 0],
                    [-1, 1],
                    [0, -1],
                    [0, 0],
                    [0, 1],
                    [1, -1],
                    [1, 0],
                    [1, 1]]


class OneCarAgent(Agent):

    def __init__(self, env):
        super().__init__(env)
        self.len_without_junc = self.env.length - JUNCTION_SIZE - 1
        self.q_agent = ApproximateQAgent(extractor=SimpleExtractor(), actionFn=lambda s: POSSIBLE_ACTIONS)
        self.last_report = None
        self.new_report = None

    def send_control_signal(self):
        if self.last_report is not None and self.new_report is not None:
            self.q_agent.update(self.last_report, self.new_report.actions,
                                self.new_report, self.env.get_score_for_round(self.new_report))
        if self.new_report is None:
            actions = [0, 0]
        else:
            actions = self.q_agent.getAction(self.new_report)  # [ , ]
        for car in self.env.cars_iterator():
            if self.is_closest_car(car) and self.last_report is not None:  # agent cars
                car.update_speed(actions[car.path])
                continue
            closest = self.get_closest_car_in_path(car)  # , self.len_without_junc - car.dist)
            if car.speed ** 2 + DECELERATION_FACTOR * closest >= -3:
                car.update_speed(-1)
            else:
                car.update_speed(1)
        self.last_report = self.new_report
        self.new_report = self.env.propagate(actions)

    def get_closest_car_in_path(self, car):
        speed_state, age_state = self.env.get_state()
        path = speed_state[car.path][car.dist + 1:]
        cars = np.where(path != -1)[0]
        if cars.__len__() == 0:
            return self.env.length * 2
        closest = cars[0]
        return closest - 3

    def is_closest_car(self, car):
        speed_state, age_state = self.env.get_state()
        path = speed_state[car.path][car.dist + 1:]
        cars = np.where(path != -1)[0]
        if cars.__len__() == 0:
            return True
        return False
