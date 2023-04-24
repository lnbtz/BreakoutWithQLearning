import random

import numpy as np
from collections import deque

from util.networkInitializer import init_q_net

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
        main_q_net = init_q_net(self.environment, self.learningRate)
        main_q_net.set_weights(self.qNet.get_weights())
        replay_memory = deque(maxlen=self.REPLAY_MEMORY_LENGTH)

        steps = 0
        for episode in range(self.numberOfGames):
            state, _ = self.environment.reset()

            done = False
            while not done:
                if random.uniform(0, 1) <= self.explorationRate:
                    action = self.environment.env.action_space.sample()
                else:
                    reshaped_state = state.reshape([1, state.shape[0]])
                    predicted_q_values = main_q_net.predict(reshaped_state).flatten()
                    action = predicted_q_values.argmax()

                new_state, reward, done = self.environment.step(action)
                steps += 1
                replay_memory.append([state, action, reward, new_state, done])

                if steps % self.BACKPROPAGATION_RATE == 0 or done:
                    self.train(replay_memory, main_q_net)

                state = new_state

            # Done!
            if steps >= self.COPY_STEP_LIMIT:
                self.qNet.set_weights(main_q_net.get_weights())
                steps = 0

            self.learningRate = self.MIN_LEARNING_RATE + (self.MAX_LEARNING_RATE - self.MAX_LEARNING_RATE) * np.exp(-self.decayRate * episode)

    def train(self, replay_memory, main_q_net):
        if len(replay_memory) < self.MIN_REPLAY_SIZE:
            return

        batch = random.sample(replay_memory, self.BATCH_SIZE)
        states = np.array([sample[0] for sample in batch])
        predicted_q_values = main_q_net.predict(states)
        new_states = np.array([sample[3] for sample in batch])
        target_q_values = self.qNet.predict(new_states)

        X = []
        Y = []
        for index, (state, action, reward, _, done) in enumerate(batch):
            if not done:
                max_future_reward = reward + self.discountFactor * np.max(target_q_values[index])
            else:
                max_future_reward = reward

            q_values = predicted_q_values[index]
            q_values[action] = (1 - self.learningRate) * q_values[action] + self.learningRate * max_future_reward

            X.append(state)
            Y.append(q_values)

        main_q_net.fit(np.array(X), np.array(Y), batch_size=self.BATCH_SIZE, verbose=0, shuffle=True)

