from Agent import Agent
from DQNModel import DQNModel
import numpy as np
import torch

# from DQNTrainer import DQNTrainer
# from NormalEnvironment import NormalEnvironment

ACCELERATION_FACTOR = 2


class DQNAgent(Agent):

    def __init__(self, env, exploration_proba, n_actions):
        super().__init__(env)
        self.exploration_proba = exploration_proba
        self.n_actions = n_actions
        self.model = self.get_model(None)


    def send_control_signal(self):
        speed_state, age_state = self.env.get_state()
        location = np.zeros((self.env.num_paths, self.env.length+1))  #todo:+1
        actions_dict = {}
        for car in self.env.cars_iterator():
            cur_state = np.array((speed_state, age_state, location))
            cur_state[2, car.path, car.dist] = 1
            action = self.compute_action(cur_state)
            car.update_speed(action * ACCELERATION_FACTOR)
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
            return np.random.choice(range(3)) - 1
        q_values = self.model(torch.tensor(current_state).unsqueeze(0).double()).detach().numpy()
        return np.argmax(q_values) - 1

    def get_model(self, path):
        model = DQNModel().double()
        if path is not None:
            model.load_state_dict(torch.load(path))
        return model
        # trainer = DQNTrainer(model, self.exploration_proba, self.n_actions)
        # trainer.train()
        # return model

    # def train_model(self):
    #     n_episodes = 500
    #     max_iterations = 4000
    #     total_steps = 0
    #     for episode in range(n_episodes):
    #         env = NormalEnvironment(4, 150)
    #         agent = DQNAgent(env, )
    #         for iteration in range(self.max_iterations):

# from NormalEnvironment import NormalEnvironment
#
# e = NormalEnvironment(4, 150)
# a = DQNAgent(e)
# a.send_control_signal()

