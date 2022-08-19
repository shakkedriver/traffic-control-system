# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import random


# import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
    """

    def __init__(self, **args):
        """You can initialize Q-values here..."""
        ReinforcementAgent.__init__(self, **args)
        self.q_values = Counter()
        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
        return self.q_values[(state, action)]

    def getValue(self, state):
        """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
        actions = self.getLegalActions(state)
        max_action = None
        max_val = -float('inf')
        for action in actions:
            cur_val = self.getQValue(state, action)
            if cur_val > max_val:
                max_val = cur_val
                max_action = action
        # if max_action is None:
        #     return max_val
        return max_val

    def getPolicy(self, state):
        """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
        actions = self.getLegalActions(state)
        if not actions:
            return None
        max_val = self.getValue(state)
        max_actions = [a for a in actions if self.getQValue(state, a) == max_val]
        if max_actions == []:
            self.getValue(state)
        return random.choice(max_actions)

    def getAction(self, state):
        """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if random.random() < self.epsilon:
            return random.choice(legalActions)
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
        new_val = self.getQValue(state, action)
        new_val += self.alpha * (
                    reward + self.discount * self.getValue(nextState) - self.getQValue(state, action))
        self.q_values[(state, action)] = new_val


class PacmanQAgent(QLearningAgent):
    """Exactly the same as QLearningAgent, but with different default parameters"""

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """

    def __init__(self, extractor, **args):
        self.featExtractor = extractor  # todo: changed
        PacmanQAgent.__init__(self, **args)
        self.weights = Counter()

        # You might want to initialize weights here.
        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        my_sum = 0
        for f in features:
            my_sum += features[f] * self.weights[f]
        return my_sum

    def update(self, state, action, nextState, reward):
        """
        Should update your weights based on transition
        """
        new_val = self.alpha * (
                    reward + self.discount * self.getValue(nextState) - self.getQValue(state, action))
        features = self.featExtractor.getFeatures(state, action)
        for f in features:
            self.weights[f] += new_val * features[f]
        print(reward, self.weights)

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
