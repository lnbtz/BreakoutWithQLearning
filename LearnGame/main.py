import gymnasium

import QLearning.qlearning

env = gymnasium.make("Taxi-v3")

QLearning.qlearning.q_learn(env)
