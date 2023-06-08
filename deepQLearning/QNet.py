from tensorflow import keras
import tensorflow as tf


class QNet:
    def __init__(self, model):
        self.model = model

    def getBestAction(self, state):
        reshaped_state = state / 255
        reshaped_state = tf.expand_dims(reshaped_state, 0)
        predicted_q_values = self.model(reshaped_state, training=False)
        action = tf.argmax(predicted_q_values[0]).numpy()
        # reshaped_state = state.reshape([1, state.shape[0]])
        # q_values = self.model.predict(reshaped_state / 255).flatten()
        return action

    def getQValues(self, state):
        reshaped_state = state / 255
        reshaped_state = tf.expand_dims(reshaped_state, 0)
        predicted_q_values = self.model(reshaped_state, training=False)

        # reshaped_state = state.reshape([1, state.shape[0]])
        # q_values = self.model.predict(reshaped_state / 255).flatten()
        return predicted_q_values[0]

    @staticmethod
    def loadFromFile(pathToModel):
        model = keras.saving.load_model(pathToModel)
        return QNet(model)
