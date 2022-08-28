# featureExtractors.py
# --------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""Feature extractors for Pacman game states"""

from Counter import Counter
from CarFactory import NormalCarFactory
import numpy as np


class FeatureExtractor:
    def getFeatures(self, state, action):
        """
        Returns a dict from features to counts
        Usually, the count will just be 1.0 for
        indicator functions.
        """
        pass


class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def __init__(self, env):
        self.env = env

    def getFeatures(self, report, action):
        length_of_path = len(report.speed_state[0])
        features = Counter()
        speed_state = report.speed_state
        path0_cars = np.where(speed_state[0] != -1)
        if len(path0_cars[0]) == 0:
            speed0 = NormalCarFactory.NORMAL_CAR_MAX_INIT_SPEED
            dist0 = length_of_path - self.env.junction_size
            time0 = dist0 / NormalCarFactory.NORMAL_CAR_MAX_INIT_SPEED
        else:
            car0 = path0_cars[0][-1]
            speed0 = max(min(speed_state[0][car0] + action[0], NormalCarFactory.NORMAL_CAR_MAX_INIT_SPEED), 0)
            dist0 = length_of_path - self.env.junction_size - car0 - speed0
            if speed0 > 0:
                time0 = dist0 / speed0
            else:
                time0 = length_of_path

        path1_cars = np.where(speed_state[1] != -1)
        if len(path1_cars[0]) == 0:
            speed1 = NormalCarFactory.NORMAL_CAR_MAX_INIT_SPEED
            dist1 = length_of_path - self.env.junction_size
            time1 = dist1 / NormalCarFactory.NORMAL_CAR_MAX_INIT_SPEED
        else:
            car1 = path1_cars[0][-1]
            speed1 = max(min(speed_state[1][car1] + action[1], NormalCarFactory.NORMAL_CAR_MAX_INIT_SPEED), 0)
            dist1 = length_of_path - self.env.junction_size - car1 - speed1
            if speed1 > 0:
                time1 = dist1 / speed1
            else:
                time1 = length_of_path
        features["speed"] = speed0 + speed1
        max_speed = max(speed1, speed0)
        min_dist = min(dist0, dist1)
        not_in_junc = dist0 < length_of_path - self.env.junction_size and dist1 < length_of_path - self.env.junction_size
        time_diff = int(not_in_junc) * min(abs(time0 - time1), self.env.junction_size * 2 / NormalCarFactory.NORMAL_CAR_MAX_INIT_SPEED)
        features["time_diff"] = time_diff
        features["bias"] = 1.0
        features["collision"] = int(-self.env.junction_size < dist0 < 0 and -self.env.junction_size < dist1 < 0)
        features.divideAll(10.0)
        return features
