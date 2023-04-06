import random


class QLearning:
    def __init__(self, environment, qTable, alpha, epsilon, gamma, numberOfGames, savingPath):
        self.environment = environment
        self.qTable = qTable
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.numberOfGames = numberOfGames
        self.savingPath = savingPath

    def qLearn(self):
        # start training
        for i in range(1, self.numberOfGames):
            # setup game
            state = self.environment.reset()
            state = self.environment.step(0)

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
                # update q table
                self.qTable.updateQValue(state, action, new_value)
                # update state
                state = next_state

                epochs += 1

            print("done after {} epochs\n".format(epochs))

        print(self.savingPath)
        self.qTable.saveToFile(self.savingPath)
        self.environment.close()
        print("done training")
