from deepQLearning.DeepQLearning import DeepQLearning
from environment.Environment import Environment
from environment.observationTransformers.StandardObservationTransformer import StandardObservationTransformer
from get_project_root import root_path
from util.networkInitializer import init_q_net_breakout, init_q_net_cartpole
from util.options import OPT_GAME_BREAKOUT, OPT_GAME_CARTPOLE


class WrongGameError(Exception):
    pass


class Config:
    def __init__(self,
                 game,
                 onlyOneLife,
                 envObsType,
                 learning_rate,
                 exploration_rate,
                 min_exploration_rate,
                 discount_factor,
                 solutionRunningReward,
                 decay_rate,
                 savingPath=None,
                 observationTransformer=StandardObservationTransformer()
                 ):
        self.onlyOneLife = onlyOneLife
        self.envObsType = envObsType
        self.learningRate = learning_rate
        self.explorationRate = exploration_rate
        self.discountFactor = discount_factor
        self.solutionRunningReward = solutionRunningReward
        self.decayRate = decay_rate
        environment = Environment(game, onlyOneLife, envObsType, observationTransformer)

        if game == OPT_GAME_BREAKOUT:
            q_net = init_q_net_breakout(environment.env, self.learningRate)
        elif game == OPT_GAME_CARTPOLE:
            q_net = init_q_net_cartpole(environment.env, self.learningRate)
        else:
            raise WrongGameError

        self.deepQLearning = DeepQLearning(environment, q_net, learning_rate, exploration_rate, min_exploration_rate, discount_factor, solutionRunningReward,
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
