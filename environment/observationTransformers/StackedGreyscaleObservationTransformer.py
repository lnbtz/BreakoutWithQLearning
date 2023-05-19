from collections import deque
import numpy as np
import tensorflow as tf


class StackedGreyscaleObservationTransformer:
    TARGET_SHAPE = (84, 84)

    def __init__(self):
        self.stack = deque(maxlen=4)
        black_array = np.zeros((210, 160))
        self.stack.append(black_array)
        self.stack.append(black_array)
        self.stack.append(black_array)
        self.stack.append(black_array)

    def transform(self, observation):
        self.stack.append(observation)
        stacked = tf.stack(self.stack, axis=-1)
        return tf.keras.preprocessing.image.smart_resize(stacked, self.TARGET_SHAPE, interpolation='bilinear') / 255


if __name__ == "__main__":
    obs_trans = StackedGreyscaleObservationTransformer()
    img = np.random.rand(210, 160)
    result = obs_trans.transform(img)
    print("")
