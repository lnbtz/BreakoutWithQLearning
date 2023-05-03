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
    EXPLORATION_FRAMES = 50_000
    MAX_STEPS_PER_EPISODE = 10_000

    loss_function = keras.losses.Huber()

    def __init__(self, environment, qNet, learning_rate, exploration_rate, min_exploration_rate, discount_factor, solutionRunningReward, decay_rate,
                 savingPath):
        self.environment = environment
        self.qNet = qNet
        self.learningRate = learning_rate
        self.explorationRate = exploration_rate
        self.minExplorationRate = min_exploration_rate
        self.discountFactor = discount_factor
        self.decayRate = decay_rate
        self.savingPath = savingPath
        self.solutionRunningReward = solutionRunningReward
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    def deepQLearn(self):
        main_q_net = init_q_net(self.environment, self.learningRate)
        main_q_net.set_weights(self.qNet.get_weights())
        replay_memory = []
        reward_memory = []
        running_reward = 0
        total_steps = 0
        episode_count = 1

        while True:
            state, _ = self.environment.reset()
            episode_reward = 0

            # No infinite Loops
            for step in range(1, self.MAX_STEPS_PER_EPISODE):
                total_steps += 1

                if total_steps < self.EXPLORATION_FRAMES or random.uniform(0, 1) <= self.explorationRate:
                    action = self.environment.env.action_space.sample()
                else:
                    reshaped_state = state.reshape([1, state.shape[0]])
                    predicted_q_values = main_q_net(tf.convert_to_tensor(reshaped_state)).numpy().flatten()
                    action = predicted_q_values.argmax()

                new_state, reward, done = self.environment.step(action)

                replay_memory.append([state, action, reward, new_state, done])
                episode_reward += reward

                if total_steps % self.BACKPROPAGATION_RATE == 0:
                    self.train(replay_memory, main_q_net)

                state = new_state

                if len(replay_memory) > self.REPLAY_MEMORY_LENGTH:
                    del replay_memory[:1]

                if total_steps % self.COPY_STEP_LIMIT == 0:
                    self.qNet.set_weights(main_q_net.get_weights())
                    print("Running Reward: " + str(running_reward) + " at Episode " + str(episode_count) + " with Epsilon " + str(self.explorationRate))

                self.explorationRate = self.minExplorationRate + (self.MAX_EXPLORATION_RATE - self.minExplorationRate) * np.exp(-self.decayRate * episode_count)

                if done:
                    break

            # Episode done
            reward_memory.append(episode_reward)
            if len(reward_memory) > 100:
                del reward_memory[:1]
            running_reward = np.mean(reward_memory)

            episode_count += 1

            # Durchschnittlicher reward der letzten 100 Episoden
            if running_reward > 20:
                print("Found Solution after " + str(episode_count) + " Episodes")
                break

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

        Y = []
        max_future_rewards = ((rewards * self.discountFactor * tf.reduce_max(target_q_values, axis=1)) * (1 - dones) - dones).numpy()
        for index in range(self.BATCH_SIZE):
            q_values = predicted_q_values[index]
            q_values[actions[index]] = max_future_rewards[index]
            Y.append(q_values)

        with tf.GradientTape() as tape:
            predicted_q_values = main_q_net(states)
            loss = self.loss_function(Y, predicted_q_values)

        grads = tape.gradient(loss, main_q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, main_q_net.trainable_variables))
