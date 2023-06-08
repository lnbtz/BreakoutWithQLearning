from environment.Environment import Environment
from environment.observationTransformers.StackedGreyscaleObservationTransformer import \
    StackedGreyscaleObservationTransformer
from environment.observationTransformers.StandardObservationTransformer import StandardObservationTransformer
from util.options import *
from get_project_root import root_path
from deepQLearning.QNet import QNet
from util.showDeepQGame import showQGame


qNets_path = root_path(ignore_cwd=True) + '/specialNets/'
file_name = "breakoutLeonKindaWorking"

qNet = QNet.loadFromFile(qNets_path + file_name)
env = Environment(OPT_GAME_BREAKOUT, False, OPT_ENV_GREYSCALE, StackedGreyscaleObservationTransformer())


# qNet = QNet.loadFromFile(qNets_path + "cartpole")
# env = Environment(OPT_GAME_CARTPOLE, False, OPT_ENV_RAM, StandardObservationTransformer())
showQGame(env, qNet)
