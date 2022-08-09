from Agent import Agent
from DQNModel import DQNModel
import numpy as np
import torch

ACCELERATION_FACTOR = 1

class DQNAgent(Agent):

    def __init__(self, env, exploration_proba, n_actions):
        super().__init__(env)
        self.exploration_proba = exploration_proba
        self.n_actions = n_actions
        self.model = DQNModel()

    def send_control_signal(self):
        speed_state, age_state = self.env.get_state()
        location = np.zeros((self.env.num_paths, self.env.length+1))  #todo:+1
        for car in self.env.cars_iterator():
            cur_state = np.array((speed_state, age_state, location))
            cur_state[car.path, car.dist, 2] = 1
            action = self.compute_action(cur_state)
            car.update_speed(action * ACCELERATION_FACTOR)

    def compute_action(self, current_state):
        # We sample a variable uniformly over [0,1]
        # if the variable is less than the exploration probability
        #     we choose an action randomly
        # else
        #     we forward the state through the DNN and choose the action
        #     with the highest Q-value.
        if np.random.uniform(0, 1) < self.exploration_proba:
            return np.random.choice(range(3)) - 1
        q_values = self.model(torch.tensor(current_state, dtype=).unsqueeze(0))
        return np.argmax(q_values)


# from NormalEnvironment import NormalEnvironment
#
# e = NormalEnvironment(4, 150)
# a = DQNAgent(e)
# a.send_control_signal()

