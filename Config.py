from qLearning.QLearning import QLearning
from Environment import Environment
from observationTransformers.StandardObservationTransformer import StandardObservationTransformer
from get_project_root import root_path
from os import path


class PathError(Exception):
    pass


class Config:
    def __init__(self,
                 onlyOneLife,
                 envObsType,
                 alpha,
                 epsilon,
                 gamma,
                 numberOfGames,
                 savingPath=None,
                 observationTransformer=StandardObservationTransformer(),
                 loadingPath=None
                 ):
        self.environment = Environment(onlyOneLife, envObsType, observationTransformer)
        self.qLearning = QLearning(self.environment, alpha, epsilon, gamma, numberOfGames, self._initSavingPath(savingPath), loadingPath)

    def doRun(self):
        self.qLearning.qLearn()

    @staticmethod
    def _initSavingPath(savingPath):
        if savingPath is None:
            savingPath = root_path(ignore_cwd=False) + '/qTables/'

        if not path.exists(savingPath):
            raise PathError
        else:
            return savingPath

