import random
from abc import ABC, abstractmethod

import numpy as np
import torch

from Car import NormalCar
from DQNModel import DQNModel
from featureExtractors import SimpleExtractor
from qlearningAgents import ApproximateQAgent

ACTION_LOOKUP = [-1, 0, 1]


class Agent(ABC):

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def send_control_signal(self):
        pass


class AcceleratingAgent(Agent):
    ACCELERATION_FACTOR = 2

    def __init__(self, env):
        super(AcceleratingAgent, self).__init__(env)

    def send_control_signal(self):
        for car in self.env.cars_iterator():
            action = self.compute_action()
            car.update_speed(ACTION_LOOKUP[action] * self.ACCELERATION_FACTOR)

    def compute_action(self):
        return 2


class DQNAgent(Agent):

    ACCELERATION_FACTOR = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, env, model=DQNModel(), exploration_proba=0, n_actions=3): # todo: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        super().__init__(env)
        self.exploration_proba = exploration_proba
        self.n_actions = n_actions
        self.model = model

    def send_control_signal(self):
        speed_state, age_state = self.env.get_state()
        location = np.zeros((self.env.num_paths, self.env.length + 1))  # todo:+1
        actions_dict = {}
        for car in self.env.cars_iterator():
            cur_state = np.array((speed_state, age_state, location))
            cur_state[2, car.path, car.dist] = 1
            action = self.compute_action(cur_state)
            car.update_speed(ACTION_LOOKUP[action] * self.ACCELERATION_FACTOR)
            actions_dict[car] = (action, car.path, car.dist)
        return actions_dict

    def compute_action(self, current_state):
        # We sample a variable uniformly over [0,1]
        # if the variable is less than the exploration probability
        #     we choose an action randomly
        # else
        #     we forward the state through the DNN and choose the action
        #     with the highest Q-value.
        if np.random.uniform(0, 1) < self.exploration_proba:
            return np.random.choice(range(3))
        q_values = self.model(
            torch.tensor(current_state).to(self.device).unsqueeze(0).double()).detach().cpu().numpy()
        return np.argmax(q_values)

    def get_model(self, path):
        model = DQNModel().double()
        if path is not None:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        return model


class LightTrafficAgent(Agent):
    LEN_OF_GREEN = 100
    DELAY_BETWEEN_GREENS = 15

    def __init__(self, env):
        super().__init__(env)
        self.cur_green_path = 0
        self.counter = 0
        self.all_red = False
        self.len_without_junc = self.env.length - self.env.junction_size - 1

    def send_control_signal(self):
        # speed_state, age_state = self.env.get_state()
        for car in self.env.cars_iterator():
            if (car.path == self.cur_green_path and not self.all_red) or car.path >= self.len_without_junc:
                closest = self.get_closest_car_in_path(car)
            else:
                closest = min(self.get_closest_car_in_path(car), self.len_without_junc - car.dist)
            if car.speed ** 2 + NormalCar.DECELERATION_FACTOR * closest >= -3:
                car.update_speed(-1)
            # elif(closest):
            else:
                car.update_speed(1)
        self.counter += 1
        if (self.counter % self.LEN_OF_GREEN) > self.LEN_OF_GREEN - self.DELAY_BETWEEN_GREENS:
            self.all_red = True
        if (self.counter % self.LEN_OF_GREEN) == 0:
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


class RandomAgent(Agent):
    ACCELERATION_FACTOR = 2

    def __init__(self, env):
        super(RandomAgent, self).__init__(env)

    def send_control_signal(self):
        for car in self.env.cars_iterator():
            action = self.compute_action()
            car.update_speed(ACTION_LOOKUP[action] * self.ACCELERATION_FACTOR)

    def compute_action(self):
        return random.choice([0, 1, 2])


class OneCarAgent(Agent):
    POSSIBLE_ACTIONS = [[-2, -2],
                        [-2, 0],
                        [-2, 1],
                        [0, -2],
                        [0, 0],
                        [0, 1],
                        [1, -2],
                        [1, 0],
                        [1, 1]]

    def __init__(self, env, numTraining=1000, weights_path='weights.pkl'):
        super().__init__(env)
        self.len_without_junc = self.env.length - self.env.junction_size - 1
        self.q_agent = ApproximateQAgent(extractor=SimpleExtractor(env), env=env, numTraining=numTraining,
                                         weights_path=weights_path, actionFn=lambda s: self.POSSIBLE_ACTIONS)
        self.last_report = None
        self.new_report = None

    def send_control_signal(self):
        if self.last_report is not None and self.new_report is not None:
            self.q_agent.update(self.last_report, self.last_report.actions,
                                self.new_report, self.env.get_score_for_round(self.new_report))
        if self.new_report is None:
            actions = [0, 0]
        else:
            actions = self.q_agent.getAction(self.new_report)  # [ , ]
        for car in self.env.cars_iterator():
            if self.is_closest_car(car) and self.last_report is not None:  # agent cars
                car.update_speed(actions[car.path])
                continue
            closest = min(self.get_closest_car_in_path(car), self.len_without_junc - car.dist)
            # closest = self.get_closest_car_in_path(car)  # , self.len_without_junc - car.dist)
            if car.speed ** 2 + NormalCar.DECELERATION_FACTOR * closest >= -3 or\
                    car.speed ** 2 + NormalCar.DECELERATION_FACTOR * self.get_closest_car_in_other_path(car) >= -3:
                car.update_speed(-1)
            else:
                car.update_speed(1)
        self.last_report = self.new_report
        self.new_report = self.env.get_last_report(actions)

    def get_closest_car_in_path(self, car):
        speed_state, age_state = self.env.get_state()
        path = speed_state[car.path][car.dist + 1:]
        cars = np.where(path != -1)[0]
        if cars.__len__() == 0:
            return self.env.length * 2
        closest = cars[0]
        return closest - 3

    def get_closest_car_in_other_path(self, car):
        speed_state, age_state = self.env.get_state()
        path = speed_state[1 - car.path][car.dist:]
        cars = np.where(path != -1)[0]
        if cars.__len__() == 0 or car.path < self.env.junction_size / 3:
            return self.env.length * 2
        closest = self.find_nearest(cars, car.path)

        return closest

    def is_closest_car(self, car):
        speed_state, age_state = self.env.get_state()
        path = speed_state[car.path][car.dist + 1:]
        cars = np.where(path != -1)[0]
        if cars.__len__() == 0:
            return True
        return False

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
