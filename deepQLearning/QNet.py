from tensorflow import keras


class QNet:
    def __init__(self, pathToModel):
        self.model = keras.saving.load_model(pathToModel)

    def getBestAction(self, state):
        reshaped_state = state.reshape([1, state.shape[0]])
        q_values = self.model.predict(reshaped_state).flatten()
        return q_values.argmax()
