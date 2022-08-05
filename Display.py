# import matplotlib.pyplot as plt
# from matplotlib import animation
# import numpy as np
#
# from NormalEnvironment import JUNCTION_SIZE
#
#
# class DisplayGUI:
#
#     def __init__(self, env, func):
#         self.env = env
#         self.func = func
#         self.fig = plt.figure()
#         self.data = self.env.get_state()[0] + 1
#         self.im = plt.imshow(self.data)
#         self.anim = animation.FuncAnimation(self.fig, lambda i: self.func(self), interval=50, frames=200)
#         self.anim.save("animation.mp4")
#         # plt.show()
#
#     def update(self, state, report):
#         state[:, -JUNCTION_SIZE:][state[:, -JUNCTION_SIZE:] == 0] = 30
#         # self.data = self.env.get_state()[0] + 1
#         self.im = plt.imshow(self.create_image(state))
#         # plt.show()
#         return self.im
#
#     @staticmethod
#     def create_image(state):
#         image = np.zeros((state.shape[0] * 5, state.shape[1]))
#         for i in range(len(state)):
#             image[i * 5:(i + 1) * 5] = state[i]
#         return image


import tkinter as tk
import numpy as np

from NormalEnvironment import JUNCTION_SIZE
from RegularReport import RegularReport

X_FACTOR = 8
Y_FACTOR = 30
CAR_HEIGHT = 1
CAR_WIDTH = 2

# if True - pauses the animation for a second when there is a collision
JUNC_COLLISION_PAUSE = False
PATH_COLLISION_PAUSE = False


class DisplayGUI:

    def __init__(self, env, func):
        self.env = env
        self.func = func
        self.root = tk.Tk()
        self.init_state = self.env.get_state()[0] + 1
        self.init_report = RegularReport([], [], [], [])
        self.main_frame = tk.Frame(self.root, height=self.init_state.shape[0], width=self.init_state.shape[1])
        self.report_label = tk.Label(self.root)
        self.update(self.init_state, self.init_report)
        self.root.mainloop()

    def update(self, state, report):
        to_sleep = 100
        self.main_frame.destroy()
        self.main_frame = tk.Frame(self.root, height=self.init_state.shape[0]*Y_FACTOR, width=self.init_state.shape[1]*X_FACTOR)

        junction_label = tk.Label(self.main_frame, bg='grey', height=self.init_state.shape[0]*Y_FACTOR, width=JUNCTION_SIZE*X_FACTOR)
        junction_label.place(y=0, x=(self.init_state.shape[1] + 1 - JUNCTION_SIZE) * X_FACTOR)

        to_sleep = self.draw_collisions(report.collisions_in_Junction, state, to_sleep, 'green', JUNC_COLLISION_PAUSE)
        to_sleep = self.draw_collisions(report.collisions_in_paths, state, to_sleep, 'blue', PATH_COLLISION_PAUSE)

        cars = np.argwhere(state != 0)
        for car in cars:
            label = tk.Label(self.main_frame, bg=self.create_color(state[car[0], car[1]] * 30), height=CAR_HEIGHT, width=CAR_WIDTH)
            label.place(y=(car[0]*Y_FACTOR), x=(car[1]*X_FACTOR))

        self.main_frame.pack(side=tk.TOP)
        # self.report_label.destroy()
        # self.report_label = tk.Label(self.root, text=str(report))
        # self.report_label.pack(side=tk.BOTTOM)
        self.root.after(to_sleep, lambda: self.func(self))

    def draw_collisions(self, lst, state, to_sleep, color, to_pause):
        for car_pair in lst:
            to_sleep = 1000 if to_pause else to_sleep
            for car in car_pair:
                x = car.dist
                y = car.path
                label = tk.Label(self.main_frame, bg=color, height=CAR_HEIGHT, width=CAR_WIDTH)
                label.place(y=(y * Y_FACTOR), x=(x * X_FACTOR))
                state[y, x] = 0
        return to_sleep

    @staticmethod
    def create_color(value):
        r = min(value, 255)
        value -= r
        g = min(value, 255)
        value -= g
        b = min(value, 255)
        return "#%02x%02x%02x" % (r, g, b)
