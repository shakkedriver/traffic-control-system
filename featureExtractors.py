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
LENGTH_OF_PATH = 150
from NormalEnvironment import JUNCTION_SIZE
from NormalCarFactory import NORMAL_CAR_MAX_INIT_SPEED
import numpy as np


class FeatureExtractor:
    def getFeatures(self, speed_state, age_state, action, car):
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

    def getFeatures(self, speed_state, age_state, speed_change, car):
        features = Counter()

        # features of me:
        car_speed = max(0, min(car.speed + speed_change, car.max_speed))
        features["my_speed"] = car_speed
        my_age = car.age + 1
        # features["my_age"] = my_age

        my_cur_dist = LENGTH_OF_PATH - JUNCTION_SIZE - car.dist - 1
        my_dist = my_cur_dist - car_speed
        features["my_dist"] = my_dist

        my_time = my_dist / car_speed
        # features["my_time"] = my_time

        # last_cars = np.where(speed_state[car.path] != -1)
        # if len(last_cars) > 1:
        #     last_car = car.dist - last_cars[-2]
        # else:
        #     last_car = car.dist
        # features["my_last_car"] = last_car

        # features of the other car:
        other_path = 1 - car.path
        other_last_cars = np.where(speed_state[other_path] != -1)
        if len(other_last_cars) == 0:
            other_speed = NORMAL_CAR_MAX_INIT_SPEED
            other_age = 0
            other_dist = LENGTH_OF_PATH - JUNCTION_SIZE
            other_time = (LENGTH_OF_PATH - JUNCTION_SIZE) / NORMAL_CAR_MAX_INIT_SPEED

            # features["other_speed"] = NORMAL_CAR_MAX_INIT_SPEED
            # features["other_age"] = 0
            # features["other_dist"] = LENGTH_OF_PATH - JUNCTION_SIZE
            # features["other_time"] = (LENGTH_OF_PATH - JUNCTION_SIZE) / NORMAL_CAR_MAX_INIT_SPEED
            # features["other_last_car"] = LENGTH_OF_PATH - JUNCTION_SIZE
        else:
            other_car = other_last_cars[-1]
            other_speed = speed_state[other_path][other_car]
            other_age = age_state[other_path][other_car] + 1

            # features["other_speed"] = speed_state[other_path][other_car]
            # features["other_age"] = age_state[other_path][other_car] + 1

            current_other_dist = LENGTH_OF_PATH - JUNCTION_SIZE - other_car - 1
            other_dist = current_other_dist - other_speed
            # features["other_dist"] = other_dist

            other_time = other_dist / other_speed
            # features["other_time"] = other_time

            # last_cars_other = np.where(speed_state[car.path] != -1)
            # if len(other_last_cars) > 1:
            #     last_car_other = other_car - last_cars_other[-2]
            # else:
            #     last_car_other = other_car
            # features["other_last_car"] = last_car_other
        if my_time < other_time:
            first_speed = car_speed
        else:
            first_speed = other_speed
        first_time = JUNCTION_SIZE // first_speed + 1
        time_diff = min(first_time, abs(other_time - my_time))

        features["time_diff"] = time_diff
        features["bias"] = 1.0
        features["collision"] = int(my_dist >= LENGTH_OF_PATH - JUNCTION_SIZE and other_dist >= LENGTH_OF_PATH - JUNCTION_SIZE)
        features.divideAll(10.0)
        return features
