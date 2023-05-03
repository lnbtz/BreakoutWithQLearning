from util.Config import Config
from util.options import *
from get_project_root import root_path

BASE_PATH = base_path = root_path(ignore_cwd=False) + "/qNets/"

# breakout
game = OPT_GAME_BREAKOUT
one_life = True
OBS = OPT_ENV_RAM
learning_rate = 0.00025
exploration_rate = 1
min_exploration_rate = 0.1
discount_factor = 0.99
solution_reward = 1
decay_rate = 0.01
folder_name = "breakout"

# cartpole
# game = OPT_GAME_CARTPOLE
# one_life = False
# OBS = OPT_ENV_RAM
# learning_rate = 0.001
# exploration_rate = 0.7
# min_exploration_rate = 0.1
# discount_factor = 0.618
# solution_reward = 400
# decay_rate = 0.01
# folder_name = "cartpole"


starter = Config(game,
                 one_life,
                 OBS,
                 learning_rate,
                 exploration_rate,
                 min_exploration_rate,
                 discount_factor,
                 solution_reward,
                 decay_rate,
                 BASE_PATH + folder_name)
starter.doRun()
