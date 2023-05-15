# nur für greyscale bilder
from time import sleep

import gymnasium as gym
from PIL import Image
import numpy as np

from environment.Environment import Environment
from environment.observationTransformers.StackedGreyscaleObservationTransformer import \
    StackedGreyscaleObservationTransformer


def cut_image(observation):
    height, width = observation.shape
    left_offset = 6
    right_offset = 6
    top_offset = 30
    bottom_offset = 18
    reduced_obs = np.zeros((height - (top_offset + bottom_offset), width - (left_offset + right_offset)),
                           dtype=np.uint8)
    for j in range(height - (top_offset + bottom_offset)):
        for k in range(width - (left_offset + right_offset)):
            reduced_obs[j, k] = observation[j + top_offset, k + left_offset]
    return reduced_obs


def compress_image(observation):
    vertical_compression = 2
    horizontal_compression = 2
    height, width = observation.shape
    assert (height / vertical_compression).is_integer()
    assert (width / horizontal_compression).is_integer()

    comperessed_height = int(height / vertical_compression)
    comperessed_width = int(width / horizontal_compression)
    comperessed_img = np.zeros((comperessed_height, comperessed_width), dtype=np.uint8)

    for j in range(0, height, vertical_compression):
        for k in range(0, width, horizontal_compression):
            max_value = max(observation[j:j + vertical_compression, k:k + horizontal_compression].flatten())
            comperessed_img[int(j / vertical_compression), int(k / horizontal_compression)] = max_value

    return comperessed_img


def combine_images(observation1, observation2):
    height, width = observation1.shape
    cumulated_img = np.zeros((height, width), dtype=np.uint8)
    for j in range(height):
        for k in range(width):
            obs1_val = observation1[j, k]
            obs2_val = observation2[j, k]
            cumulated_img[j, k] = max(obs2_val, obs1_val)

    return cumulated_img


def show_img(observation):
    Image.fromarray(observation, "L").show()


if __name__ == "__main__":
    env = Environment("BreakoutDeterministic-v4", True, "grayscale", StackedGreyscaleObservationTransformer(84, 84, (210, 160)))
    # env = gym.make("BreakoutDeterministic-v4", render_mode="rgb_array", obs_type="grayscale")
    env.reset()

    observation, _, _ = env.step(1)
    for i in range(4):
        observation, _, _ = env.step(1)



    # show_img(observation)
    #
    # cut_img = cut_image(observation)
    # show_img(cut_img)
    #
    # compressed_img = compress_image(cut_img)
    # show_img(compressed_img)
    #
    # combined = combine_images(observation1, observation2)
    # combined_cut = cut_image(combined)
    # show_img(combined_cut)
    # combined_compressed = compress_image(combined_cut)

    img1 = np.zeros((84, 84), dtype=np.uint8)
    img2 = np.zeros((84, 84), dtype=np.uint8)
    img3 = np.zeros((84, 84), dtype=np.uint8)
    img4 = np.zeros((84, 84), dtype=np.uint8)
    for i in range(84):
        for j in range(84):
            img1[i][j] = observation[i][j][0]
            img2[i][j] = observation[i][j][1]
            img3[i][j] = observation[i][j][2]
            img4[i][j] = observation[i][j][3]

show_img(img1)

show_img(img2)

show_img(img3)

show_img(img4)
# show_img(combined_compressed)
