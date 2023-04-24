from util.Config import Config
from util.options import *
from get_project_root import root_path

BASE_PATH = base_path = root_path(ignore_cwd=False) + "/qNets/"

result_folder_name = BASE_PATH + "test"
starter = Config(True, OPT_ENV_RAM, 0.7, 1, 0.6, 50, 0.01, result_folder_name)
starter.doRun()
