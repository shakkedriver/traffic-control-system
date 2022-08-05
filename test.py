from NormalEnvironment import NormalEnvironment
from Display import DisplayGUI


USE_DISPLAY = True

def main(disp):
    r = env.propagate()
    e = env.get_state()[0]
    # print(env.get_state()[0])
    print(r)
    disp.update(e + 1, r)


if __name__ == '__main__':
    env = NormalEnvironment(10, 150)
    if not USE_DISPLAY:
        while True:
            r = env.propagate()
            e = env.get_state()[0]
            #print(env.get_state()[0])
            print(r)
    displayer = DisplayGUI(env, main)
