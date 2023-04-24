from deepQLearning.DeepQLearning import DeepQLearning
from environment.Environment import Environment
from environment.observationTransformers.StandardObservationTransformer import StandardObservationTransformer
from get_project_root import root_path
from util.networkInitializer import init_q_net


class Config:
    def __init__(self,
                 onlyOneLife,
                 envObsType,
                 learning_rate,
                 exploration_rate,
                 discount_factor,
                 numberOfGames,
                 decay_rate,
                 savingPath=None,
                 observationTransformer=StandardObservationTransformer()
                 ):
        self.onlyOneLife = onlyOneLife
        self.envObsType = envObsType
        self.learningRate = learning_rate
        self.explorationRate = exploration_rate
        self.discountFactor = discount_factor
        self.numberOfGames = numberOfGames
        self.decayRate = decay_rate
        environment = Environment(onlyOneLife, envObsType, observationTransformer)
        q_net = init_q_net(environment.env, self.learningRate)
        self.deepQLearning = DeepQLearning(environment, q_net, learning_rate, exploration_rate, discount_factor, numberOfGames,
                                           self.decayRate,
                                           self._initSavingPath(savingPath))

    def doRun(self):
        self.deepQLearning.deepQLearn()

    @staticmethod
    def _initSavingPath(savingPath):
        if savingPath is None:
            return root_path(ignore_cwd=False) + '/qNets/' + "default"
        else:
            return savingPath
