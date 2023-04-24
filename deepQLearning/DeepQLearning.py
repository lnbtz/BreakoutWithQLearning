import random


class DeepQLearning:
    def __init__(self, environment, qNet, learning_rate, exploration_rate, discount_factor, numberOfGames, decay_rate, savingPath):
        self.environment = environment
        self.qNet = qNet
        self.learningRate = learning_rate
        self.explorationRate = exploration_rate
        self.discountFactor = discount_factor
        self.decayRate = decay_rate
        self.numberOfGames = numberOfGames
        self.savingPath = savingPath


    def deepQLearn(self):
        for i in range(self.numberOfGames):
            state, _ = self.environment.reset()
            print(state)
            