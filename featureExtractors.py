# featureExtractors.py
# --------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""Feature extractors for Pacman game states"""

# from game import Directions, Actions
# import util
from Counter import Counter
from NormalEnvironment import LENGTH_OF_PATH
from NormalEnvironment import JUNCTION_SIZE
from NormalCarFactory import NORMAL_CAR_MAX_INIT_SPEED
import numpy as np


class FeatureExtractor:
    def getFeatures(self, state, action):
        """
        Returns a dict from features to counts
        Usually, the count will just be 1.0 for
        indicator functions.
        """
        pass


class IdentityExtractor(FeatureExtractor):
    pass
    # def getFeatures(self, speed_state, age_state, action):
    #     feats = Counter()
    #     feats[(state, action)] = 1.0
    #     return feats


# def closestFood(pos, food, walls):
#     """
#   closestFood -- this is similar to the function that we have
#   worked on in the search project; here its all in one place
#   """
#     fringe = [(pos[0], pos[1], 0)]
#     expanded = set()
#     while fringe:
#         pos_x, pos_y, dist = fringe.pop(0)
#         if (pos_x, pos_y) in expanded:
#             continue
#         expanded.add((pos_x, pos_y))
#         # if we find a food at this location then exit
#         if food[pos_x][pos_y]:
#             return dist
#         # otherwise spread out from the location to its neighbours
#         nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
#         for nbr_x, nbr_y in nbrs:
#             fringe.append((nbr_x, nbr_y, dist + 1))
#     # no food found
#     return None


class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, report, action):
        features = Counter()
        speed_state = report.speed_state
        path0_cars = np.where(speed_state[0] != -1)
        if len(path0_cars[0]) == 0:
            speed0 = NORMAL_CAR_MAX_INIT_SPEED
            dist0 = LENGTH_OF_PATH - JUNCTION_SIZE
            # features["dist_0"] = dist0
            time0 = dist0 / NORMAL_CAR_MAX_INIT_SPEED
            # features["age_0"] = 0
        else:
            car0 = path0_cars[0][-1]
            speed0 = max(min(speed_state[0][car0] + action[0], NORMAL_CAR_MAX_INIT_SPEED), 0)
            dist0 = LENGTH_OF_PATH - JUNCTION_SIZE - car0 - speed0
            # features["dist_0"] = dist0
            if speed0 > 0:
                time0 = dist0 / speed0
            else:
                time0 = JUNCTION_SIZE
            # features["age_0"] = report.age_state[0][car0] / 50

        path1_cars = np.where(speed_state[1] != -1)
        if len(path1_cars[0]) == 0:
            speed1 = NORMAL_CAR_MAX_INIT_SPEED
            dist1 = LENGTH_OF_PATH - JUNCTION_SIZE
            # features["dist_1"] = dist1
            time1 = dist1 / NORMAL_CAR_MAX_INIT_SPEED
            # features["age_1"] = 0
        else:
            car1 = path1_cars[0][-1]
            speed1 = max(min(speed_state[1][car1] + action[1], NORMAL_CAR_MAX_INIT_SPEED), 0)
            dist1 = LENGTH_OF_PATH - JUNCTION_SIZE - car1 - speed1
            #       LENGTH_OF_PATH - JUNCTION_SIZE < car.dist + features["speed"]
            # features["dist_1"] = dist1
            if speed1 > 0:
                time1 = dist1 / speed1
            else:
                time1 = JUNCTION_SIZE
            # features["age_1"] = report.age_state[1][car1] / 50
        # features["speed_0"] = speed0
        # features["speed_1"] = speed1
        features["speed"] = speed0 + speed1
        # if time0 < time1:
        #     first_speed = speed0
        #     first_time = time0
        # else:
        #     first_speed = speed1
        #     first_time = time1
        # first_time = JUNCTION_SIZE // (first_speed + 0.1) + 1
        time_diff = min(JUNCTION_SIZE, abs(time0 - time1))

        features["time_diff"] = time_diff
        features["bias"] = 1.0
        features["collision"] = int(-JUNCTION_SIZE < dist0 < 0 and -JUNCTION_SIZE < dist1 < 0)
        features.divideAll(10.0)
        return features
