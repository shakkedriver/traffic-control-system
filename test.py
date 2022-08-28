import json
import random
import numpy as np
import torch
from torch.distributed.elastic import agent
import argparse

from DQNModel import DQNModel
from JunctionEnvironment import NormalEnvironment
from Display import DisplayGUI
from Agent import AcceleratingAgent, DQNAgent, LightTrafficAgent, RandomAgent, OneCarAgent


def lookup(name, namespace):
    """
    Get a method or class from any imported module from its name.
    Usage: lookup(functionName, globals())
    """
    dots = name.count('.')
    if dots > 0:
        moduleName, objName = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
        module = __import__(moduleName)
        return getattr(module, objName)
    else:
        modules = [obj for obj in namespace.values() if str(type(obj)) == "<type 'module'>"]
        options = [getattr(module, name) for module in modules if name in dir(module)]
        options += [obj[1] for obj in namespace.items() if obj[0] == name]
        if len(options) == 1: return options[0]
        if len(options) > 1: raise Exception('Name conflict for %s')
        raise Exception('%s not found as a method or class' % name)


def get_model(path):
    model = DQNModel().double()
    if path is not None:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model


def main(disp):
    agent.send_control_signal()
    r = env.propagate()
    e = env.get_state()[0]
    disp.update(e + 1, r)


if __name__ == '__main__':
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--display', action='store_true', help="show visual simulation")
    parser.add_argument('-a', '--agent', type=str, required=True, help="choose which agent to use",
                        choices=['AcceleratingAgent', 'DQNAgent', 'LightTrafficAgent', 'RandomAgent', 'OneCarAgent'])
    parser.add_argument('-c', '--config', type=str, required=True, help="configuration file path")
    parser.add_argument('-m', '--model', type=str, help="path for a trained DQN model, required for 'DQNAgent'")
    parser.add_argument('-n', '--numTraining', type=int, help="number of training iteration for OneCarAgent")
    parser.add_argument('-p', '--weightsPath', type=str, help="path for a weights file for OneCarAgent")

    args = parser.parse_args()

    if args.agent == 'DQNAgent' and args.model is None:
        parser.error('For running a DQNAgent, path for a trained model must be specify')
        exit()
    user_agent = lookup(args.agent, globals())

    with open(args.config, 'r') as f:
        params = json.load(f)
    env = NormalEnvironment(2, 100, params)  # todo: leave it as constants?
    # print(env.params)
    if args.agent == 'DQNAgent':
        # todo: DQNAgent(env, 0, 3, get_model(...)) - can the "0" and "3" remain as constants?
        agent = user_agent(env, get_model(args.model))
    elif args.agent == 'OneCarAgent':
        num_training = args.numTraining if args.numTraining is not None else 1000
        if args.weightsPath:
            agent = user_agent(env, num_training, args.weightsPath)
        else:
            agent = user_agent(env, num_training)
    else:
        agent = user_agent(env)

    # print(agent)
    # agent = DQNAgent(env, 0, 3,get_model(ttt))
    # agent = LightTrafficAgent(env)
    # agent = LightTrafficAgent(env)
    # agent = RandomAgent(env)
    # agent = AcceleratingAgent(env)
    if not args.display:  # todo: what happens?
        while True:
            agent.send_control_signal()
            r = env.propagate()
            e = env.get_state()[0]
            print(env.get_state()[0])
            print(r)
    else:
        displayer = DisplayGUI(env, main)
