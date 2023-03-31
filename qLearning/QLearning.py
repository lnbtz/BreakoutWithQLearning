import numpy as np
import random
from QTable import QTable


class QLearning:
    def __init__(self, environment, alpha, epsilon, gamma, numberOfGames, pathToExisting=None):
        self.environment = environment
        self.qTable = QTable(environment.observationSpaceSize, environment.actionSpaceSize, pathToExisting)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.numberOfGames = numberOfGames

    def qLearn(self):
        # start training
        for i in range(1, self.numberOfGames):
            # setup game
            state = self.environment.reset()
            epochs = 0
            done = False

            # play game
            while not done:

                if random.uniform(0, 1) < self.epsilon:
                    action = random.randrange(self.environment.actionSpaceSize)
                else:
                    action = self.qTable.getBestAction(state)  # don't explore and take best known move

                # get next state
                next_state, reward, done = self.environment.step(action)

                # get last value
                old_value = self.qTable.getQValue(state, action)
                # get next best value
                next_max = self.qTable.getNextMax(next_state)
                # get new value
                new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                # https://miro.medium.com/v2/resize:fit:1072/format:webp/1*y0V_OFDJIcamdP7kCw7v5Q.png
                # https://towardsdatascience.com/q-learning-algorithm-from-explanation-to-implementation-cdbeda2ea187
                # update q table
                self.qTable.updateQValue(state, action, new_value)
                # update state
                state = next_state

                epochs += 1

            print("done after {} epochs\n".format(epochs))

        self.qTable.saveToFile("newFile.h5")
        self.environment.close()
        print("done training")
