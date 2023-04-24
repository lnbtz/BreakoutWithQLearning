from tensorflow import keras


class QNet:
    def __init__(self, model):
        self.model = model

    def getBestAction(self, state):
        reshaped_state = state.reshape([1, state.shape[0]])
        q_values = self.model.predict(reshaped_state).flatten()
        return q_values.argmax()

    @staticmethod
    def loadFromFile(pathToModel):
        model = keras.saving.load_model(pathToModel)
        return QNet(model)
