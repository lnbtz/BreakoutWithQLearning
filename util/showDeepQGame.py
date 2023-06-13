import sys

import imageio
import numpy as np
import pygame
import tensorflow as tf
from util.ui.showFilters import *

from matplotlib import pyplot as plt
from pygame.locals import *

PLAY_SPEED = 50
DELAY_AFTER_DEATH = 1  # Seconds
SIZE_SCALE = 5
SCREEN_X = 1920
SCREEN_Y = 1080


def showQGame(env, qNet, file_name, auto_shoot):
    conv_layer_index = [0, 1, 2, 3]
    outputs = [qNet.model.layers[i].output for i in conv_layer_index]
    model_short = Model(inputs=qNet.model.inputs, outputs=outputs)
    print(model_short.summary())

    # Create the figure and axes
    fig, ax = plt.subplots()

    bar_names = ["NOOP", "FIRE", "RIGHT", "LEFT"]
    # Create four initial data points
    numbers = [0, 0, 0, 0]

    # Create the bar plot
    bars = ax.bar(range(4), numbers)

    # Set the axis labels and title
    ax.set_xlabel("Actions")
    ax.set_ylabel("Q-Value")
    ax.set_title("Real-time Q-Value Visualization")

    # Set the x-axis tick labels
    ax.set_xticks(range(4))
    ax.set_xticklabels(bar_names)

    # Set the initial y-axis limits
    ax.set_ylim(0, 1)

    frames = []

    # Initialize pygame
    observation = env.reset()

    ary = env.render()
    width = SCREEN_X
    height = SCREEN_Y

    pygame.init()
    display = pygame.display.set_mode((width, height), FULLSCREEN)
    pygame.display.set_caption(env.game + ' - Q-Agent')
    clock = pygame.time.Clock()

    total_rewards = 0
    ball_dropped = False
    is_paused = False
    running = True
    if auto_shoot:
        env.step(1)
    while running:
        if not is_paused:
            if ball_dropped and auto_shoot:
                env.step(1)

            action = qNet.getBestAction(observation)
            observation, reward, terminated, ball_dropped = env.step(action)
            total_rewards += reward

            ary = env.render()
            mutate_image(ary, display)
            pygame.display.update()
            mutate_frames(display, frames)

            clock.tick(PLAY_SPEED)
            if terminated:
                observation, total_rewards = terminate(env, frames, observation, total_rewards, file_name)
                env.close()
                running = False
        else:
            clock.tick(PLAY_SPEED)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    observation, total_rewards = terminate(env, frames, observation, total_rewards, file_name)
                    env.close()
                    running = False
                if event.key == pygame.K_ESCAPE:
                    sys.exit(0)
                if event.key == pygame.K_RIGHT:
                    actions = qNet.getQValues(observation)
                    numpy_actions = actions.numpy()
                    print("Q-Values: ")
                    print("NOOP: " + str(numpy_actions[0])
                          + " FIRE: " + str(numpy_actions[1])
                          + " RIGHT: " + str(numpy_actions[2])
                          + " LEFT: " + str(numpy_actions[3]))
                    normalized_numbers = (numpy_actions - np.min(numpy_actions)) / (
                                np.max(numpy_actions) - np.min(numpy_actions))
                    numbers = normalized_numbers

                    for bar, number in zip(bars, numbers):
                        bar.set_height(number)
                    fig.canvas.draw()
                    plt.pause(0.001)

                    action = tf.argmax(actions).numpy()
                    observation, reward, terminated, ball_dropped = env.step(action)
                    if ball_dropped:
                        env.step(1)
                    total_rewards += reward

                    ary = env.render()
                    mutate_image(ary, display)
                    pygame.display.update()
                    mutate_frames(display, frames)

                    clock.tick(PLAY_SPEED)

                    if terminated:
                        observation, total_rewards = terminate(env, frames, observation, total_rewards, file_name)
                if event.key == pygame.K_SPACE:
                    is_paused = not is_paused
            if event.type == pygame.QUIT:
                observation, total_rewards = terminate(env, frames, observation, total_rewards, file_name)
                env.close()
                running = False


def terminate(env, frames, observation, total_rewards, qNetName):
    observation = env.reset()
    print("Total Reward: " + str(total_rewards))
    total_rewards = 0
    pygame.time.wait(DELAY_AFTER_DEATH * 1000)
    #save_gif(frames, qNetName)
    return observation, total_rewards


def mutate_image(ary, display):
    img = pygame.surfarray.make_surface(ary)
    img = pygame.transform.rotate(img, -90)
    img = pygame.transform.flip(img, True, False)
    # resize the image so the entire screen is bigger with the same information
    img = pygame.transform.scale(img, (len(ary[0]) * SIZE_SCALE, len(ary) * SIZE_SCALE))
    display.blit(img, (SCREEN_X / 2 - len(ary[0]) * SIZE_SCALE / 2, SCREEN_Y / 2 - len(ary) * SIZE_SCALE / 2))


def mutate_frames(display, frames):
    frame = pygame.surfarray.array3d(display)
    frame = np.rot90(frame)
    frame = np.rot90(frame)
    frame = np.rot90(frame)
    frame = np.flip(frame, 1)
    frames.append(frame)


def save_gif(frames, qNetName):
    # Define the output file path
    output_file = "/Users/leonbeitz/PycharmProjects/BreakoutWithQLearning/qNets/breakoutProgressGifs/" + qNetName + ".gif"

    # Save the frames as a GIF
    imageio.mimsave(output_file, np.array(frames), duration=20)
