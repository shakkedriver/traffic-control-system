import math
from collections import deque

from NormalEnvironment import NormalEnvironment
from DQNAgent import DQNAgent
import numpy as np
import torch
from DQNModel import DQNModel
import torch.nn as nn
import torch.optim as optim
import tqdm
ROOT_GDRIVE_PATH = "/content/drive/MyDrive/"
GDRIVE_SAVE_REL_PATH = "ai project/"
FULL_GDRIVE_SAVE_PATH = ROOT_GDRIVE_PATH + GDRIVE_SAVE_REL_PATH
path = lambda x: FULL_GDRIVE_SAVE_PATH+ x
BATCH_SIZE = 128
GAMMA = 0.95
EPS_END = 1
EPS_START = 0
EPS_DECAY = 750
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device}")


class DQNTrainer:

    def __init__(self, nn, n_actions, n_episodes=500, max_iterations=1000):
        self.n_episodes = n_episodes
        self.max_iterations = max_iterations
        self.exploration_proba = 1
        self.n_actions = n_actions

        self.policy_net = nn.to(device).double()
        self.target_net = DQNModel().to(device).double()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(int(1e6))

    def train(self):
        total_steps = 0
        for episode in tqdm.tqdm(range(self.n_episodes)):
            env = NormalEnvironment(2, 150)
            self.exploration_proba = EPS_START + (EPS_END-EPS_START) * math.exp(-1. * episode / EPS_DECAY)
            agent = DQNAgent(env, self.exploration_proba, self.n_actions,self.policy_net)
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


                if total_steps >= BATCH_SIZE:
                    self.optimize_model(iteration % 4 == 0)

            print(f"\nscore : {score}")
            print(f"eps : {self.exploration_proba}")
            torch.save(self.policy_net.state_dict(), path("PATH"))

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
        if len(self.memory) < BATCH_SIZE:
            return
        batch = self.memory.sample()

        state_batch = torch.stack([torch.tensor(d["cur_state"]) for d in batch]).to(device).double()
        action_batch = torch.stack([torch.tensor([d["action"]]) for d in batch]).to(device)
        reward_batch = torch.stack([torch.tensor(d["reward"]) for d in batch]).to(device).double()
        next_state_batch = torch.stack([torch.tensor(d["next_state"]) for d in batch]).to(device).double()
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if copy_to_target:
            self.target_net.load_state_dict(self.policy_net.state_dict())


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, record):
        """Save a transition"""
        self.memory.append(record)

    def sample(self):
        return np.random.choice(self.memory, BATCH_SIZE, replace=False)

    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    t = DQNTrainer(DQNModel(), 3, 2400 * 3, 1000)
    t.train()
