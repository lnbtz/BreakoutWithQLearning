import h5py
import numpy as np


class QTable:
    def __init__(self, observationSpace, actionSpace, pathToExisting=None):
        self.observationSpace = observationSpace
        self.actionSpace = actionSpace

        if pathToExisting is None:
            self.qTable = np.zeros([observationSpace, actionSpace])
        else:
            with h5py.File(pathToExisting, "r") as f:
                self.qTable = f["q_table"][:]

    def getBestAction(self, state):
        return np.argmax(self.qTable[state])
    
    def getQValue(self, state, action):
        return self.qTable[state, action]

    def updateQValue(self, state, action, newValue):
        self.qTable[state, action] = newValue

    def getNextMax(self, nextState):
        np.max(self.qTable[nextState])

    def saveToFile(self, path):
        with h5py.File(path, "w") as f:
            f.create_dataset("q_table", data=self.qTable)
