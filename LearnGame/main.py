import gymnasium

import qLearning.QLearning as qLearning

env = gymnasium.make("Taxi-v3")
qLearning.q_learn(env)
