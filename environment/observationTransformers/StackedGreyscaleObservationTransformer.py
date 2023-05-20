from collections import deque
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class StackedGreyscaleObservationTransformer:
    TARGET_SHAPE = (84, 84)

    def __init__(self):
        self.stack = deque(maxlen=4)
        black_array = np.zeros((150, 160))
        self.stack.append(black_array)
        self.stack.append(black_array)
        self.stack.append(black_array)
        self.stack.append(black_array)

    def transform(self, observation):
        self.stack.append(observation[50:200])
        stacked = tf.stack(self.stack, axis=-1)
        transformed_state = tf.keras.preprocessing.image.smart_resize(stacked, self.TARGET_SHAPE, interpolation='bilinear') / 255
        # self._show_state(transformed_state)
        return transformed_state

    def _show_state(self, transformedState):
        stack = transformedState * 255
        stack = tf.unstack(stack, axis=2)
        stack = list(map(lambda frame: frame.numpy(), stack))
        plt.imshow(stack[0], cmap="gray")
        plt.imshow(stack[1], cmap="gray")
        plt.imshow(stack[2], cmap="gray")
        plt.imshow(stack[3], cmap="gray")
        plt.show()


if __name__ == "__main__":
    obs_trans = StackedGreyscaleObservationTransformer()
    img = np.random.rand(210, 160)
    result = obs_trans.transform(img)
    print("")
