from NormalEnvironment import NormalEnvironment
from DQNAgent import DQNAgent
import numpy as np


class DQNTrainer:

    def __init__(self, nn, exploration_proba, n_actions, n_episodes=500, max_iterations=4000):
        self.nn = nn
        self.n_episodes = n_episodes
        self.max_iterations = max_iterations
        self.exploration_proba = exploration_proba
        self.n_actions = n_actions

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
