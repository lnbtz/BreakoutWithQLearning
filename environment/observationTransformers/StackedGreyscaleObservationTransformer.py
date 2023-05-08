import time
from collections import deque
import numpy as np
from util.imageProcessing import *


class StackedGreyscaleObservationTransformer:
    def __init__(self, observationShape, compressionHeight, compressionWidth):
        self.observationShape = observationShape
        self.compressionHeight = compressionHeight
        self.compressionWidth = compressionWidth
        self.stack = deque(maxlen=4)
        black_array = np.zeros(observationShape)
        black_array_transformed = self._transform(black_array)
        self.stack.append(black_array_transformed)
        self.stack.append(black_array_transformed)
        self.stack.append(black_array_transformed)
        self.stack.append(black_array_transformed)

    def _transform(self, observation):
        startTime = time.time_ns
        img = cut_img(observation)
        endTime = time.time_ns()
        print( "millisecs cut_img: " + str((endTime - startTime) / 1000000000))
        return compress_image(cut_image(observation))

    def transform(self, observation):
        transformed_observation = self._transform(observation)
        self.stack.append(transformed_observation)
        return np.stack(self.stack, axis=-1)
        # result = np.zeros((transformed_observation.shape[0], transformed_observation.shape[1], 4))

        # for i in range(transformed_observation.shape[0]):
        #     for j in range(transformed_observation.shape[1]):
        #         for img_index in range(len(self.stack)):
        #             result[i][j][img_index] = self.stack[img_index][i][j]

        # return result


if __name__=="__main__":
    obs_trans = StackedGreyscaleObservationTransformer((210, 160), 2, 2)
    img = np.zeros((210, 160))
    result = obs_trans.transform(img)
    print("")

