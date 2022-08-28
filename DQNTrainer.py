import argparse
import json
import math
from collections import deque


import numpy as np
import torch

from Agent import DQNAgent
from DQNModel import DQNModel
import torch.nn as nn
import torch.optim as optim
import tqdm

from JunctionEnvironment import NormalEnvironment

# ROOT_GDRIVE_PATH = "/content/drive/MyDrive/"
# GDRIVE_SAVE_REL_PATH = "ai project/"
# FULL_GDRIVE_SAVE_PATH = ROOT_GDRIVE_PATH + GDRIVE_SAVE_REL_PATH
# path = lambda x: FULL_GDRIVE_SAVE_PATH + x
# todo : don't need?? @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# EPS_END = 1
# EPS_START = 0
# EPS_DECAY = 750
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device}")


class DQNTrainer:

    def __init__(self, nn, n_actions, n_episodes, max_iterations, params, my_path, batch_size, gamma):
        self.n_episodes = n_episodes
        self.max_iterations = max_iterations
        self.exploration_proba = 0
        self.n_actions = n_actions
        self.params = params
        self.my_path = my_path
        self.batch_size = batch_size
        self.gamma = gamma

        self.policy_net = nn.to(device).double()
        self.target_net = DQNModel().to(device).double()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(int(1e6), self.batch_size)

    def train(self):
        total_steps = 0
        for episode in tqdm.tqdm(range(self.n_episodes)):
            env = NormalEnvironment(2, self.params['path_length'], self.params)
            # self.exploration_proba = EPS_START + (EPS_END-EPS_START) * math.exp(-1. * episode / EPS_DECAY)
            agent = DQNAgent(env, self.policy_net, self.exploration_proba, self.n_actions)
            score = 0
            for iteration in range(self.max_iterations):
                total_steps += 1
                cur_speed, cur_age = env.get_state()
                actions_dict = agent.send_control_signal()
                report = env.propagate()
                reward = env.get_score_for_round(report)
                new_speed, new_age = env.get_state()
                done = False  # todo: change?
                location = np.zeros((env.num_paths, env.length + 1))
                for car in actions_dict:
                    self.memory.push(self.create_records(car, location, cur_speed, cur_age,
                                                         actions_dict[car][1], actions_dict[car][2],
                                                         reward, actions_dict[car][0], new_speed,
                                                         new_age, done))
                score += reward

                # if total_steps >= batch_size:
                #     agent.train(batch_size=batch_size)

                if total_steps >= self.batch_size:
                    self.optimize_model(iteration % 4 == 0)

            print(f"\nscore : {score}")
            print(f"eps : {self.exploration_proba}")
            torch.save(self.policy_net.state_dict(), self.my_path)

    def create_records(self, car, location, cur_speed, cur_age, cur_path, cur_dist, reward, action, new_speed, new_age,
                       done):
        d = {}
        cur_state = np.array((cur_speed, cur_age, location))
        cur_state[2, cur_path, cur_dist] = 1
        next_state = np.array((new_speed, new_age, location))
        if car.dist < location.shape[1]:
            next_state[2, car.path, car.dist] = 1

        d["cur_state"] = cur_state
        d["next_state"] = next_state
        d["action"] = action
        d["reward"] = reward
        d["done"] = done

        return d

    def optimize_model(self, copy_to_target=True):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample()

        state_batch = torch.stack([torch.tensor(d["cur_state"]) for d in batch]).to(device).double()
        action_batch = torch.stack([torch.tensor([d["action"]]) for d in batch]).to(device)
        reward_batch = torch.stack([torch.tensor(d["reward"]) for d in batch]).to(device).double()
        next_state_batch = torch.stack([torch.tensor(d["next_state"]) for d in batch]).to(device).double()
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if copy_to_target:
            self.target_net.load_state_dict(self.policy_net.state_dict())


class ReplayMemory(object):

    def __init__(self, capacity, batch_size):
        self.memory = deque([], maxlen=capacity)
        self.batch_size = batch_size

    def push(self, record):
        """Save a transition"""
        self.memory.append(record)

    def sample(self):
        return np.random.choice(self.memory, self.batch_size, replace=False)

    def __len__(self):
        return len(self.memory)


def assign_default_value(val, default):
    return val if val is not None else default


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, required=True, help="configuration file path")
    parser.add_argument('-p', '--path', type=str, required=True, help="path to save the trained model")
    parser.add_argument('--n_actions', type=int, help="number of possible actions")
    parser.add_argument('--n_episodes', type=int, help="number of episodes in the training process")
    parser.add_argument('--max_iterations', type=int, help="number of iterations per episode")
    parser.add_argument('-b', '--batch_size', type=int, help="batch size for training")
    parser.add_argument('-g', '--gamma', type=float, help="gamma param for training")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        params = json.load(f)
    n_actions = assign_default_value(args.n_actions, 3)
    n_episodes = assign_default_value(args.n_episodes, 100)
    max_iterations = assign_default_value(args.max_iterations, 100)
    batch_size = assign_default_value(args.batch_size, 128)
    gamma = assign_default_value(args.gamma, 0.95)

    t = DQNTrainer(DQNModel(), n_actions, n_episodes, max_iterations, params, args.path, batch_size, gamma)
    t.train()
