from util.Config import Config
from util.options import *
from get_project_root import root_path

BASE_PATH = base_path = root_path(ignore_cwd=False) + "/qNets/"


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
