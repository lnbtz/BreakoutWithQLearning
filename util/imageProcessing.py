# nur für greyscale bilder

import gymnasium as gym
from PIL import Image
import numpy as np


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
    assert (height/vertical_compression).is_integer()
    assert (width/horizontal_compression).is_integer()

    comperessed_height = int(height / vertical_compression)
    comperessed_width = int(width / horizontal_compression)
    comperessed_img = np.zeros((comperessed_height, comperessed_width), dtype=np.uint8)

    for j in range(0, height, vertical_compression):
        for k in range(0, width, horizontal_compression):
            max_value = max(observation[j:j+vertical_compression, k:k+horizontal_compression].flatten())
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

if __name__=="__main__":
    env = gym.make("BreakoutDeterministic-v4", render_mode="rgb_array", obs_type="grayscale")
    env.reset()

    observation, _, _, _, _ = env.step(1)
    observation1, _, _, _, _ = env.step(0)
    observation2, _, _, _, _ = env.step(0)

    show_img(observation)

    cut_img = cut_image(observation)
    show_img(cut_img)

    compressed_img = compress_image(cut_img)
    show_img(compressed_img)

    combined = combine_images(observation1, observation2)
    combined_cut = cut_image(combined)
    show_img(combined_cut)
    combined_compressed = compress_image(combined_cut)
    show_img(combined_compressed)