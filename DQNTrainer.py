from NormalEnvironment import NormalEnvironment
from DQNAgent import DQNAgent
import numpy as np
import torch
from DQNModel import DQNModel
import torch.nn as nn
import torch.optim as optim

BATCH_SIZE = 128
GAMMA = 0.95
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNTrainer:

    def __init__(self, nn, exploration_proba, n_actions, n_episodes=500, max_iterations=1000):
        self.n_episodes = n_episodes
        self.max_iterations = max_iterations
        self.exploration_proba = exploration_proba
        self.n_actions = n_actions

        self.policy_net = nn.to(device).double()
        self.target_net = DQNModel().to(device).double()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

    def train(self):
        total_steps = 0
        for episode in range(self.n_episodes):
            actions_lst = []
            env = NormalEnvironment(4, 150)
            agent = DQNAgent(env, self.exploration_proba, self.n_actions)
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
                    actions_lst.append(self.create_records(car, location, cur_speed, cur_age,
                                                           actions_dict[car][1], actions_dict[car][2],
                                                           reward, actions_dict[car][0], new_speed,
                                                           new_age, done))
            # todo: sample randomly

            # if total_steps >= batch_size:
            #     agent.train(batch_size=batch_size)
            if total_steps >= BATCH_SIZE:
                self.optimize_model(actions_lst)
    def create_records(self, car, location, cur_speed, cur_age, cur_path, cur_dist, reward, action, new_speed, new_age, done):
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

    def optimize_model(self, memory):
        if len(memory) < BATCH_SIZE:
            return
        batch = np.random.choice(memory, BATCH_SIZE, replace=False)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        # batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                         batch.next_state)), device=device, dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state
        #                                    if s is not None])
        state_batch = torch.stack([torch.tensor(d["cur_state"]) for d in batch]).to(device).double()
        action_batch = torch.stack([torch.tensor([d["action"]]) for d in batch]).to(device)
        reward_batch = torch.stack([torch.tensor(d["reward"])for d in batch]).to(device).double()
        next_state_batch = torch.stack([torch.tensor(d["next_state"]) for d in batch]).to(device).double()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        # next_state_values = torch.zeros((BATCH_SIZE,), device=device)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
if __name__ == '__main__':
    t = DQNTrainer(DQNModel(),0.1,3)
    t.train()