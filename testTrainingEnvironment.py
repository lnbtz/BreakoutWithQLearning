from TrainingEnvironment import TrainingEnvironment
from MockQ import MockQ
from options import *

trainingEnv = TrainingEnvironment(MockQ(), False, True, OPT_ENV_RAM)
trainingEnv.doTraining(1000)
