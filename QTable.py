import h5py
import numpy as np
import pickle
from datetime import datetime


class QTable:
    def __init__(self, actionSpace, pathToExisting=None):
        self.actionSpace = actionSpace

        if pathToExisting is None:
            self.qTable = {}
        else:
            with h5py.File(pathToExisting, "r") as f:
                self.qTable = f["q_table"][:]

    def getBestAction(self, state):
        max_value = -np.inf
        best_action = None
        for i in range(self.actionSpace):
            state_value_hash = (str(state), i)  #String value
            try:
                q_value = self.qTable[state_value_hash]
            except KeyError:
                q_value = 0
            if q_value > max_value:
                max_value = q_value
                best_action = i
        return best_action

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
        pickle.dump(self.qTable, open(path + datetime.now().strftime("%m|%d|%Y|%H:%M:%S") + '.qtable', "wb"))
