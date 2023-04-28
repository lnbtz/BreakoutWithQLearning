import random

import numpy as np
from tensorflow import keras
import tensorflow as tf
from util.networkInitializer import init_q_net


class DeepQLearning:
    BACKPROPAGATION_RATE = 4
    REPLAY_MEMORY_LENGTH = 50_000
    MIN_REPLAY_SIZE = 1000
    BATCH_SIZE = 64 * 2
    COPY_STEP_LIMIT = 100
    MAX_EXPLORATION_RATE = 1
    MIN_EXPLORATION_RATE = 0.01
    EXPLORATION_FRAMES = 50_000

    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    loss_function = keras.losses.Huber()

    def __init__(self, environment, qNet, learning_rate, exploration_rate, discount_factor, numberOfGames, decay_rate,
                 savingPath):
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
        replay_memory = []

        steps = 0
        frames_until_no_random = 0
        for episode in range(self.numberOfGames):
            print("Episode: " + str(episode+1))
            state, _ = self.environment.reset()

            done = False
            while not done:
                if frames_until_no_random < self.EXPLORATION_FRAMES or random.uniform(0, 1) <= self.explorationRate:
                    action = self.environment.env.action_space.sample()
                else:
                    reshaped_state = state.reshape([1, state.shape[0]])
                    predicted_q_values = main_q_net.call(tf.convert_to_tensor(reshaped_state)).numpy().flatten()
                    action = predicted_q_values.argmax()

                new_state, reward, done = self.environment.step(action)
                steps += 1
                replay_memory.append([state, action, reward, new_state, done])

                if steps % self.BACKPROPAGATION_RATE == 0 or done:
                    self.train(replay_memory, main_q_net)

                state = new_state

                if len(replay_memory) > self.REPLAY_MEMORY_LENGTH:
                    del replay_memory[:1]

            # Done!
            if steps >= self.COPY_STEP_LIMIT:
                print("Copying weights to Target Net")
                self.qNet.set_weights(main_q_net.get_weights())
                steps = 0

            frames_until_no_random += 1
            self.explorationRate = self.MIN_EXPLORATION_RATE + (self.MAX_EXPLORATION_RATE - self.MAX_EXPLORATION_RATE) * np.exp(-self.decayRate * episode)

        print("Saving Target Net")
        keras.saving.save_model(self.qNet, self.savingPath)

    def train(self, replay_memory, main_q_net):
        if len(replay_memory) < self.MIN_REPLAY_SIZE:
            return

        random_indices = np.random.choice(range(len(replay_memory)), size=self.BATCH_SIZE)
        states = np.array([replay_memory[i][0] for i in random_indices])
        predicted_q_values = main_q_net(tf.convert_to_tensor(states)).numpy()
        new_states = np.array([replay_memory[i][3] for i in random_indices])
        target_q_values = self.qNet(tf.convert_to_tensor(new_states)).numpy()
        rewards = np.array([replay_memory[i][2] for i in random_indices])
        dones = np.array([float(replay_memory[i][4]) for i in random_indices])
        actions = np.array([replay_memory[i][1] for i in random_indices])

        X = []
        Y = []
        for index in range(self.BATCH_SIZE):
            if not dones[index]:
                max_future_reward = rewards[index] + self.discountFactor * np.max(target_q_values[index])
            else:
                max_future_reward = -1

            q_values = predicted_q_values[index]
            q_values[actions[index]] = (1 - self.learningRate) * q_values[actions[index]] + self.learningRate * max_future_reward

            X.append(states[index])
            Y.append(q_values)

        with tf.GradientTape() as tape:
            predicted_q_values = main_q_net(states)
            loss = self.loss_function(Y, predicted_q_values)

        grads = tape.gradient(loss, main_q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, main_q_net.trainable_variables))
