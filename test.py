from NormalEnvironment import NormalEnvironment
if __name__ == '__main__':

    env = NormalEnvironment(50,150)
    while True:
        r=env.propagate()
        e = env.get_state()[0]
        # print(env.get_state()[0])
        print(r)