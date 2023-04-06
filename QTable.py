import numpy as np
import pickle
from datetime import datetime
import random


class QTable:
    def __init__(self, actionSpace, trainingParameter):
        self.actionSpace = actionSpace
        self.qTable = {}
        self.trainingParameter = trainingParameter

    def getBestAction(self, state):
        max_value = -np.inf
        best_action = None
        for i in range(self.actionSpace):
            state_value_hash = (str(state), i)  # String value
            try:
                q_value = self.qTable[state_value_hash]
            except KeyError:
                q_value = 0
            if q_value > max_value:
                max_value = q_value
                best_action = i

        return best_action if max_value != 0 else random.randrange(self.actionSpace)

    # TODO return 0 or None?
    def getQValue(self, state, action):
        try:
            q_value = self.qTable[(str(state), action)]
        except KeyError:
            q_value = 0
        return q_value

    def updateQValue(self, state, action, newValue):
        self.qTable[(str(state), action)] = newValue

    def getNextMax(self, nextState):
        max_value = -np.inf
        for i in range(self.actionSpace):
            state_value_hash = (str(nextState), i)
            try:
                q_value = self.qTable[state_value_hash]
            except KeyError:
                q_value = 0
            if q_value > max_value:
                max_value = q_value
        return max_value

    def saveToFile(self, path):
        pickle.dump(self, open(path + datetime.now().strftime("%m|%d|%Y|%H:%M:%S") + '.qtable', "wb"))

    @staticmethod
    def loadFromFile(path):
        with open(path, "r") as inp:
            return pickle.load(inp)
