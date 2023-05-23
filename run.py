from util.Config import Config
from util.options import *
from get_project_root import root_path
from environment.observationTransformers.StackedGreyscaleObservationTransformer import StackedGreyscaleObservationTransformer
from environment.observationTransformers.StandardObservationTransformer import StandardObservationTransformer
import os

BASE_PATH = base_path = os.path.join(root_path(ignore_cwd=False), "qNets")

# breakout
game = OPT_GAME_BREAKOUT
one_life = True
OBS = OPT_ENV_GREYSCALE
learning_rate = 0.00025
exploration_rate = 1
min_exploration_rate = 0.1
discount_factor = 0.99
solution_reward = 10
decay_rate = 0.999999
folder_name = "breakout"
obs_transformer = StackedGreyscaleObservationTransformer()

# cartpole
# game = OPT_GAME_CARTPOLE
# one_life = False
# OBS = OPT_ENV_RAM
# learning_rate = 0.0001
# exploration_rate = 1
# min_exploration_rate = 0.05
# discount_factor = 0.618
# solution_reward = 500
# decay_rate = 0.99999
# folder_name = "cartpole"
# obs_transformer = StandardObservationTransformer()


starter = Config(game,
                 one_life,
                 OBS,
                 learning_rate,
                 exploration_rate,
                 min_exploration_rate,
                 discount_factor,
                 solution_reward,
                 decay_rate,
                 os.path.join(BASE_PATH, folder_name),
                 obs_transformer
                 )
starter.doRun()
