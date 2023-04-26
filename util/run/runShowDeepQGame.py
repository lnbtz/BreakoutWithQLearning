from environment.Environment import Environment
from environment.observationTransformers.StandardObservationTransformer import StandardObservationTransformer
from util.options import *
from get_project_root import root_path
from deepQLearning.QNet import QNet
from util.showDeepQGame import showQGame

qNets_path = root_path(ignore_cwd=True) + '/qNets/'
file_name = "test"

qNet = QNet.loadFromFile(qNets_path + file_name)
env = Environment(True, OPT_ENV_RAM, StandardObservationTransformer())

showQGame(env, qNet)