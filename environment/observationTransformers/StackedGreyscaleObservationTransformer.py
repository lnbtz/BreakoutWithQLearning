from collections import deque

import tensorflow as tf
import numpy as np
# from util.imageProcessing import *


class StackedGreyscaleObservationTransformer:
    def __init__(self, frame_height, frame_width, observationShape):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.stack = deque(maxlen=4)
        black_array = np.zeros(observationShape)
        black_array_transformed = self._transform(black_array)
        self.stack.append(black_array_transformed)
        self.stack.append(black_array_transformed)
        self.stack.append(black_array_transformed)
        self.stack.append(black_array_transformed)

    def _transform(self, observation):
        observation = observation.reshape((observation.shape[0], observation.shape[1], 1))
        processed = tf.image.crop_to_bounding_box(observation, 34, 0, 160, 160)
        processed = tf.image.resize(processed,
                                    [self.frame_height, self.frame_width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return processed.numpy().reshape((self.frame_height, self.frame_width))

    def transform(self, observation):
        transformed_observation = self._transform(observation)
        self.stack.append(transformed_observation)
        return np.stack(self.stack, axis=-1)


if __name__ == "__main__":
    obs_trans = StackedGreyscaleObservationTransformer(84, 84, (210, 160))
    img = np.zeros((210, 160))

    result = obs_trans.transform(img)
    print(result.shape)
