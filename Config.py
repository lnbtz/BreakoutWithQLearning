from qLearning.QLearning import QLearning
from Environment import Environment
from observationTransformers.StandardObservationTransformer import StandardObservationTransformer
from get_project_root import root_path
from os import path
from QTable import QTable


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
        if loadingPath:
            q_table = QTable.loadFromFile(loadingPath)
            training_parameter = q_table.trainingParameter
            environment = Environment(training_parameter.onlyOneLife, training_parameter.envObsType,
                                      training_parameter.observationTransformer)
        else:
            training_parameter = (
                onlyOneLife,
                envObsType,
                alpha,
                epsilon,
                gamma,
                observationTransformer
            )
            environment = Environment(onlyOneLife, envObsType, observationTransformer)
            q_table = QTable(environment.actionSpaceSize, training_parameter)

        self.qLearning = QLearning(environment, q_table, alpha, epsilon, gamma, numberOfGames,
                                   self._initSavingPath(savingPath))

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
