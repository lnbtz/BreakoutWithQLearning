import random
import os

import numpy as np
from tensorflow import keras
import tensorflow as tf
from util.networkInitializer import init_q_net


class DeepQLearning:
    BACKPROPAGATION_RATE = 4
    REPLAY_MEMORY_LENGTH = 30_000
    BATCH_SIZE = 32
    COPY_STEP_LIMIT = 10_000
    MAX_EXPLORATION_RATE = 1
    EXPLORATION_FRAMES = REPLAY_MEMORY_LENGTH
    MAX_STEPS_PER_EPISODE = 10_000
    EPSILON_GREEDY_FRAMES = 1_000_000.0

    loss_function = keras.losses.Huber()

    def __init__(self, environment,
                 q_net,
                 learning_rate,
                 exploration_rate,
                 min_exploration_rate,
                 discount_factor,
                 solution_running_reward,
                 decay_rate,
                 backpropagation_rate,
                 replay_memory_length,
                 batch_size,
                 copy_step_limit,
                 max_exploration_rate,
                 exploration_frames,
                 max_steps_per_episode,
                 epsilon_greedy_frames,
                 saving_path):
        self.environment = environment
        self.qNet = q_net
        self.learningRate = learning_rate
        self.explorationRate = exploration_rate
        self.minExplorationRate = min_exploration_rate
        self.discountFactor = discount_factor
        self.decayRate = decay_rate
        self.backpropagation_rate = backpropagation_rate
        self.replay_memory_length = replay_memory_length
        self.batch_size = batch_size
        self.copy_step_limit = copy_step_limit
        self.max_exploration_rate = max_exploration_rate
        self.exploration_frames = exploration_frames
        self.max_steps_per_episode = max_steps_per_episode
        self.epsilon_greedy_frames = epsilon_greedy_frames

        self.savingPath = saving_path
        if os.path.exists(os.path.join(saving_path, "log")):
            os.remove(os.path.join(saving_path, "log"))

        self.solutionRunningReward = solution_running_reward
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    def deepQLearn(self):
        main_q_net = init_q_net(self.environment, self.learningRate)
        main_q_net.set_weights(self.qNet.get_weights())
        replay_memory = []
        reward_memory = []
        running_reward = 0
        total_steps = 0
        episode_count = 1
        best_running_reward = 0

        while True:
            state = self.environment.reset()
            episode_reward = 0

            # No infinite Loops
            for step in range(1, self.max_steps_per_episode):
                total_steps += 1

                if total_steps < self.exploration_frames or random.uniform(0, 1) <= self.explorationRate:
                    action = self.environment.env.action_space.sample()
                else:
                    reshaped_state = state / 255
                    reshaped_state = tf.expand_dims(reshaped_state, 0)
                    predicted_q_values = main_q_net(reshaped_state, training=False)
                    action = tf.argmax(predicted_q_values[0]).numpy()

                new_state, reward, done, ball_dropped = self.environment.step(action)

                replay_memory.append([state, action, reward, new_state, ball_dropped])
                episode_reward += reward

                if total_steps % self.backpropagation_rate == 0:
                    self.train(replay_memory, main_q_net)

                state = new_state

                if len(replay_memory) > self.replay_memory_length:
                    del replay_memory[:1]

                if total_steps % self.copy_step_limit == 0:
                    self.qNet.set_weights(main_q_net.get_weights())
                    print("Running Reward: " + str(running_reward) + " at Step " + str(
                        total_steps) + " with Epsilon " + str(self.explorationRate))
                    self.log(running_reward, total_steps)
                    if running_reward > best_running_reward:
                        best_running_reward = running_reward
                        keras.saving.save_model(self.qNet, self.savingPath)
                        print("New Highscore! Saving Net")

                # self.explorationRate = self.minExplorationRate + (self.MAX_EXPLORATION_RATE - self.minExplorationRate) * np.exp(-self.decayRate * episode_count)
                # self.explorationRate = max(self.minExplorationRate, self.explorationRate * self.decayRate)
                self.explorationRate -= (self.max_exploration_rate - self.minExplorationRate) / self.epsilon_greedy_frames
                self.explorationRate = max(self.explorationRate, self.minExplorationRate)

                if done:
                    break

            # Episode done
            reward_memory.append(episode_reward)
            if len(reward_memory) > 100:
                del reward_memory[:1]
            running_reward = np.mean(reward_memory)

            episode_count += 1

            # Durchschnittlicher reward der letzten 100 Episoden
            if running_reward > self.solutionRunningReward:
                print("Found Solution after " + str(episode_count) + " Episodes")
                break

        print("Saving Target Net")
        keras.saving.save_model(self.qNet, self.savingPath)

    def train(self, replay_memory, main_q_net):
        if len(replay_memory) < self.batch_size:
            return

        random_indices = np.random.choice(range(len(replay_memory)), size=self.batch_size)
        states = np.array([replay_memory[i][0] for i in random_indices])
        # predicted_q_values = main_q_net(states / 255).numpy()
        new_states = np.array([replay_memory[i][3] for i in random_indices])
        target_q_values = self.qNet(new_states / 255).numpy()
        rewards = np.array([replay_memory[i][2] for i in random_indices])
        ball_dropped = np.array([float(replay_memory[i][4]) for i in random_indices])
        actions = np.array([replay_memory[i][1] for i in random_indices])

        # Y = []
        expected_q_value = rewards + self.discountFactor * tf.reduce_max(target_q_values, axis=1)
        expected_q_value = (expected_q_value * (1 - ball_dropped) - ball_dropped).numpy()

        masks = tf.one_hot(actions, 4)

        with tf.GradientTape() as tape:
            predicted_q_values = main_q_net(states / 255)
            q_action = tf.reduce_sum(tf.multiply(predicted_q_values, masks), axis=1)
            loss = self.loss_function(expected_q_value, q_action)

        grads = tape.gradient(loss, main_q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, main_q_net.trainable_variables))

    def log(self, running_reward, total_steps):
        with open(os.path.join(self.savingPath, "log"), "a") as log:
            log.write(str(total_steps) + " " + str(running_reward) + " " + str(self.explorationRate) + "\n")
