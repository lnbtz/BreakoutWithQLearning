import random
import os
import pygame

import bz2
import pickle
import _pickle as cPickle

import numpy as np
from tensorflow import keras
import tensorflow as tf
from util.networkInitializer import init_q_net


class DeepQLearning:
    BACKPROPAGATION_RATE = 4
    REPLAY_MEMORY_LENGTH = 50_000
    BATCH_SIZE = 32
    COPY_STEP_LIMIT = 10_000
    MAX_EXPLORATION_RATE = 1
    EXPLORATION_FRAMES = REPLAY_MEMORY_LENGTH
    MAX_STEPS_PER_EPISODE = 10_000
    EPSILON_GREEDY_FRAMES = 1_000_000.0

    loss_function = keras.losses.Huber()

    def __init__(self,
                 environment,
                 qNet,
                 learning_rate,
                 exploration_rate,
                 min_exploration_rate,
                 discount_factor,
                 solutionRunningReward,
                 decay_rate,
                 qnet_saving_path,
                 visual_learning=False,
                 learning_episodes=None,
                 replay_memory_saving_path=None):

        self.environment = environment
        self.qNet = qNet
        self.learningRate = learning_rate
        self.explorationRate = exploration_rate
        self.minExplorationRate = min_exploration_rate
        self.discountFactor = discount_factor
        self.decayRate = decay_rate

        self.qnet_saving_path = qnet_saving_path
        if os.path.exists(os.path.join(qnet_saving_path, "log")):
            os.remove(os.path.join(qnet_saving_path, "log"))

        self.solutionRunningReward = solutionRunningReward
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

        self.visual_learning = visual_learning
        self.learning_episodes = learning_episodes
        self.replay_memory_saving_path = replay_memory_saving_path

    def deepQLearn(self):
        main_q_net = init_q_net(self.environment, self.learningRate)
        main_q_net.set_weights(self.qNet.get_weights())
        replay_memory = []
        if self.replay_memory_saving_path:
            replay_memory = self.decompress_pickle("replayMemory/running_replay_memory.pbz2")

        reward_memory = []
        running_reward = 0
        total_steps = 0
        episode_count = 1
        best_running_reward = 0

        while True:
            state = self.environment.reset()

            if self.visual_learning:
                ary = self.environment.env.render()
                width = len(ary[0])
                height = len(ary)
                pygame.init()
                display = pygame.display.set_mode((width, height))
                pygame.display.set_caption(self.environment.game + ' - Q-Agent')
                clock = pygame.time.Clock()

            episode_reward = 0

            # No infinite Loops
            for step in range(1, self.MAX_STEPS_PER_EPISODE):
                total_steps += 1

                if total_steps < self.EXPLORATION_FRAMES or random.uniform(0, 1) <= self.explorationRate:
                    action = self.environment.env.action_space.sample()
                else:
                    reshaped_state = state / 255
                    reshaped_state = tf.expand_dims(reshaped_state, 0)
                    predicted_q_values = main_q_net(reshaped_state, training=False)
                    action = tf.argmax(predicted_q_values[0]).numpy()

                new_state, reward, done, ball_dropped = self.environment.step(action)

                if self.visual_learning:
                    ary = self.environment.env.render()
                    img = pygame.surfarray.make_surface(ary)
                    img = pygame.transform.rotate(img, -90)
                    img = pygame.transform.flip(img, True, False)
                    display.blit(img, (0, 0))
                    pygame.display.update()
                    clock.tick(50)

                replay_memory.append([state, action, reward, new_state, ball_dropped])
                episode_reward += reward

                if total_steps % self.BACKPROPAGATION_RATE == 0:
                    self.train(replay_memory, main_q_net)

                state = new_state

                if len(replay_memory) > self.REPLAY_MEMORY_LENGTH:
                    del replay_memory[:1]

                if total_steps % self.COPY_STEP_LIMIT == 0:
                    self.qNet.set_weights(main_q_net.get_weights())
                    print("Running Reward: " + str(running_reward) + " at Step " + str(
                        total_steps) + " with Epsilon " + str(self.explorationRate))
                    self.log(running_reward, total_steps)
                    if running_reward > best_running_reward:
                        best_running_reward = running_reward
                        print("New Highscore! Saving Net")
                        keras.saving.save_model(self.qNet, self.qnet_saving_path)
                        # replay_memory speicher
                        self.compressed_pickle("replayMemory/running_replay_memory", replay_memory)

                # self.explorationRate = self.minExplorationRate + (self.MAX_EXPLORATION_RATE - self.minExplorationRate) * np.exp(-self.decayRate * episode_count)
                # self.explorationRate = max(self.minExplorationRate, self.explorationRate * self.decayRate)
                self.explorationRate -= (
                                                    self.MAX_EXPLORATION_RATE - self.minExplorationRate) / self.EPSILON_GREEDY_FRAMES
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

            if self.learning_episodes is not None and self.learning_episodes < episode_count:
                break

        print("Saving Target Net")
        keras.saving.save_model(self.qNet, self.qnet_saving_path)
        self.compressed_pickle("replayMemory/replay_memory_reward_" + str(running_reward), replay_memory)

    def train(self, replay_memory, main_q_net):
        if len(replay_memory) < self.BATCH_SIZE:
            return

        random_indices = np.random.choice(range(len(replay_memory)), size=self.BATCH_SIZE)
        states = np.array([replay_memory[i][0] for i in random_indices])
        new_states = np.array([replay_memory[i][3] for i in random_indices])
        target_q_values = self.qNet(new_states / 255).numpy()
        rewards = np.array([replay_memory[i][2] for i in random_indices])
        ball_dropped = np.array([float(replay_memory[i][4]) for i in random_indices])
        actions = np.array([replay_memory[i][1] for i in random_indices])

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
        with open(os.path.join(self.qnet_saving_path, "log"), "a") as log:
            log.write(str(total_steps) + " " + str(running_reward) + " " + str(self.explorationRate) + "\n")

    def compressed_pickle(self, filename, data):
        with bz2.BZ2File(filename + '.pbz2', 'w') as f:
            cPickle.dump(data, f)

    def decompress_pickle(self, file):
        data = bz2.BZ2File(file, 'rb')
        data = cPickle.load(data)
        return data
