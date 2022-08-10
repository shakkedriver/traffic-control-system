import random

import numpy as np
import torch
from torch.distributed.elastic import agent

from NormalEnvironment import NormalEnvironment
from Display import DisplayGUI
from DQNAgent import DQNAgent
from LightTrafficAgent import LightTrafficAgent
from RandomAgent import RandomAgent
from AcceleratingAgent import AcceleratingAgent

USE_DISPLAY = False

def main(disp):
    agent.send_control_signal()
    r = env.propagate()
    e = env.get_state()[0]
    # print(env.get_state()[0])
    # print(r)
    disp.update(e + 1, r)


if __name__ == '__main__':
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)

    env = NormalEnvironment(4, 150)
    agent = DQNAgent(env, 0, 0)
    # agent = LightTrafficAgent(env)
    # agent = LightTrafficAgent(env)
    # agent = RandomAgent(env)
    # agent = AcceleratingAgent(env)
    if not USE_DISPLAY:
        in_path = 0
        in_inters =0
        late = 0
        for i in range(500):
            agent.send_control_signal()
            r = env.propagate()
            in_path+=len(r.collisions_in_paths)
            in_inters+=len(r.collisions_in_Junction)
            late+=len(r.late_cars)
            e = env.get_state()[0]
            #print(env.get_state()[0])
            # print(r)
        print(f"{in_path=}, {in_inters=}, {late=}")
    # displayer = DisplayGUI(env, main)
#