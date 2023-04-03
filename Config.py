from qLearning.QLearning import QLearning
from Environment import Environment
from observationTransformers.StandardObservationTransformer import StandardObservationTransformer


class Config:
    def __init__(self,
                 onlyOneLife,
                 envObsType,
                 alpha,
                 epsilon,
                 gamma,
                 numberOfGames,
                 pathToExisting=None,
                 observationTransformer=StandardObservationTransformer()
                 ):
        self.environment = Environment(onlyOneLife, envObsType, observationTransformer)
        self.qLearning = QLearning(self.environment, alpha, epsilon, gamma, numberOfGames, pathToExisting)

    def doRun(self):
        self.qLearning.qLearn()

