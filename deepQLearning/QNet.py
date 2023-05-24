from tensorflow import keras
import tensorflow as tf


class QNet:
    def __init__(self, model):
        self.model = model

    def getBestAction(self, state):
        reshaped_state = state / 255
        reshaped_state = tf.expand_dims(reshaped_state, 0)
        q_values = self.model(reshaped_state, training=False)
        return tf.argmax(q_values[0]).numpy()

    @staticmethod
    def loadFromFile(pathToModel):
        model = keras.saving.load_model(pathToModel)
        return QNet(model)
