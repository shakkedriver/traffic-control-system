import tkinter as tk
import numpy as np
# from datetime import datetime as dt
# import os
# from PIL import ImageGrab

from Report import RegularReport

X_FACTOR = 8
Y_FACTOR = 6
CAR_HEIGHT = 1
CAR_WIDTH = 2

# if True - pauses the animation for a second when there is a collision
JUNC_COLLISION_PAUSE = False
PATH_COLLISION_PAUSE = False

# SAVE_VIDEO = True
# MANUAL = False
TO_CENTER = True


class DisplayGUI:

    def __init__(self, env, func):
        self.env = env
        self.func = func
        self.root = tk.Tk()
        self.root.geometry("810x610-100-100")
        self.init_state = self.env.get_state()[0] + 1
        self.init_report = RegularReport([], [], [], [], np.array([]), np.array([]), [])
        self.main_frame = tk.Frame(self.root, height=self.init_state.shape[0], width=self.init_state.shape[1])
        self.report_label = tk.Label(self.root)
        # if SAVE_VIDEO:
        #     self.img_dir = dt.now().strftime("%d_%m_%H_%M")
        #     os.mkdir(self.img_dir)
        #     self.frame_counter = 0
        self.update(self.init_state, self.init_report)
        self.root.mainloop()

    def update(self, state, report):
        to_sleep = 100
        self.main_frame.destroy()
        self.main_frame = tk.Frame(self.root, height=self.init_state.shape[1]*Y_FACTOR,
                                   width=self.init_state.shape[1]*X_FACTOR, bg="darkgrey")

        background_label = tk.Label(self.main_frame, bg='green',
                                    height=int(self.init_state.shape[1] - self.env.junction_size*4.4)-1,
                                    width=int((self.init_state.shape[1] - self.env.junction_size*1.8) * X_FACTOR / Y_FACTOR))
        background_label.place(x=0, y=0)
        junction_label = tk.Label(self.main_frame, bg='grey', height=self.env.junction_size*Y_FACTOR, width=self.env.junction_size*X_FACTOR)
        junction_label.place(y=(self.init_state.shape[1] + 1 - self.env.junction_size) * Y_FACTOR, x=(self.init_state.shape[1] + 1 - self.env.junction_size) * X_FACTOR)

        to_sleep = self.draw_collisions(report.collisions_in_Junction, state, to_sleep, 'green', JUNC_COLLISION_PAUSE)
        to_sleep = self.draw_collisions(report.collisions_in_paths, state, to_sleep, 'blue', PATH_COLLISION_PAUSE)

        cars = np.argwhere(state != 0)
        for car in cars:
            label = tk.Label(self.main_frame, bg=self.create_color(state[car[0], car[1]] * 30), height=CAR_HEIGHT, width=CAR_WIDTH)
            self.place_label(car, label)

        self.main_frame.pack(side=tk.TOP)
        # if SAVE_VIDEO:
        #     self.save_frame()

        # if MANUAL:
        #     self.report_label.destroy()
        #     self.report_label = tk.Button(self.root, text="Continue", command=lambda: self.func(self))
        #     self.report_label.pack(side=tk.BOTTOM)
        #     self.root.bind("<space>", lambda e: self.func(self))
        # else:
        self.root.after(to_sleep, lambda: self.func(self))

    def draw_collisions(self, lst, state, to_sleep, color, to_pause):
        for car_pair in lst:
            to_sleep = 1000 if to_pause else to_sleep
            for car in car_pair:
                x = car.dist
                y = car.path
                label = tk.Label(self.main_frame, bg=color, height=CAR_HEIGHT, width=CAR_WIDTH)
                self.place_label([car.path, car.dist], label)
                state[y, x] = 0
        return to_sleep

    def place_label(self, car, label):
        shift = self.env.junction_size // 2 - 1 if TO_CENTER else 0
        if car[0] == 1:
            label.place(y=(car[0] * Y_FACTOR) + (self.init_state.shape[1] - self.env.junction_size + shift) * Y_FACTOR,
                        x=(car[1] * X_FACTOR))
        else:
            label.place(x=(car[0] * X_FACTOR) + (self.init_state.shape[1] - self.env.junction_size + 1 + shift) * X_FACTOR,
                        y=(car[1] * Y_FACTOR))

    @staticmethod
    def create_color(value):
        r = min(value, 255)
        value -= r
        g = min(value, 255)
        value -= g
        b = min(value, 255)
        return "#%02x%02x%02x" % (r, g, b)

#     def save_frame(self):
#         widget = self.main_frame
#         x = self.root.winfo_rootx() + widget.winfo_x()
#         y = self.root.winfo_rooty() + widget.winfo_y()
#         x1 = x + widget.winfo_reqwidth()
#         y1 = y + widget.winfo_reqheight()
#         ImageGrab.grab().crop((x, y, x1, y1)).save(f"{self.img_dir}/img{self.frame_counter}.png")
#         self.frame_counter += 1
#
#
# def make_video(directory, filename):
#     import imageio
#     with imageio.get_writer(filename, mode='I') as writer:
#         files = os.listdir(directory)
#         for i in range(2, len(files)):
#             image = imageio.v2.imread(f"{directory}/img{i}.png")
#             writer.append_data(image)
#
#
# if __name__ == '__main__':
#     make_video("accelerate", "always_accelerate2.mp4")
