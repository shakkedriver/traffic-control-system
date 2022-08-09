from NormalEnvironment import NormalEnvironment
from Display import DisplayGUI
from DQNAgent import DQNAgent
from LightTrafficAgent import LightTrafficAgent

USE_DISPLAY = True

def main(disp):
    agent.send_control_signal()
    r = env.propagate()
    e = env.get_state()[0]
    # print(env.get_state()[0])
    print(r)
    disp.update(e + 1, r)


if __name__ == '__main__':
    env = NormalEnvironment(4, 150)
    # agent = DQNAgent(env, 0, 0)
    agent = LightTrafficAgent(env)
    if not USE_DISPLAY:
        while True:
            r = env.propagate()
            e = env.get_state()[0]
            #print(env.get_state()[0])
            print(r)
    displayer = DisplayGUI(env, main)
