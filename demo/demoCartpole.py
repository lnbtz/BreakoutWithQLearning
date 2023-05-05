from environment.Environment import Environment
from environment.observationTransformers.StandardObservationTransformer import StandardObservationTransformer
from util.options import *
from get_project_root import root_path
from deepQLearning.QNet import QNet
from util.showDeepQGame import showQGame

qNets_path = root_path(ignore_cwd=True) + '/specialNets/'

qNet = QNet.loadFromFile(qNets_path + "cartpole")
env = Environment(OPT_GAME_CARTPOLE, False, OPT_ENV_RAM, StandardObservationTransformer())
showQGame(env, qNet)
