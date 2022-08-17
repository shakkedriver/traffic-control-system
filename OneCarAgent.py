from Agent import Agent
import numpy as np
from NormalEnvironment import JUNCTION_SIZE
from NormalCar import DECELERATION_FACTOR
# LEN_OF_GREEN = 100
# DELAY_BETWEEN_GREENS = 15
from featureExtractors import SimpleExtractor
from qlearningAgents import ApproximateQAgent


class OneCarAgent(Agent):


    def __init__(self, env):
        super().__init__(env)
        self.len_without_junc = self.env.length - JUNCTION_SIZE - 1
        self.q_agent = ApproximateQAgent(SimpleExtractor())
        self.last_report = None

    def send_control_signal(self):
        # reward = state.getScore() - self.lastState.getScore()
        # self.observeTransition(self.lastState, self.lastAction, state, reward)
        actions = {}
        for car in self.env.cars_iterator():
            if self.is_closest_car(car) and self.last_report is not None:  # agent cars
                new_speed, new_age = self.env.get_state()
                self.q_agent.update(self.last_report.speed_state, self.last_report.age_state, self.last_report.actions[car],
                                    new_speed, new_age, self.env.get_score_for_round(self.last_report))
                action = self.q_agent.getAction(speed_state, age_state)
                actions[car] = action
                car.update_speed(action)
                continue
            closest = self.get_closest_car_in_path(car)  # , self.len_without_junc - car.dist)
            if car.speed ** 2 + DECELERATION_FACTOR * closest >= -3:
                car.update_speed(-1)
            # elif(closest):
            else:
                car.update_speed(1)
        self.last_report = self.env.propegate(actions)
        # self.counter += 1
        # if (self.counter % LEN_OF_GREEN) > LEN_OF_GREEN - DELAY_BETWEEN_GREENS:
        #     self.all_red = True
        # if (self.counter % LEN_OF_GREEN) == 0:
        #     self.all_red = False
        #     self.cur_green_path = (self.cur_green_path + 1) % self.env.num_paths

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
