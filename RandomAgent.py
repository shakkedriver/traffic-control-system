import random
from Agent import Agent
ACCELERATION_FACTOR = 2 #todo
ACTION_LOOKUP = [-1,0,1]
class RandomAgent(Agent):

    def __init__(self, env):
        super(RandomAgent, self).__init__(env)

    def send_control_signal(self):
        for car in self.env.cars_iterator():
            action = self.compute_action()
            car.update_speed(ACTION_LOOKUP[action] * ACCELERATION_FACTOR)

    def compute_action(self):
        return random.choice([0,1,2])