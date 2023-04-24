from deepQLearning.DeepQLearning import DeepQLearning
from environment.Environment import Environment
from environment.observationTransformers.StandardObservationTransformer import StandardObservationTransformer
from get_project_root import root_path
from os import path
import tensorflow as tf
from tensorflow import keras


class PathError(Exception):
    pass


class Config:
    def __init__(self,
                 onlyOneLife,
                 envObsType,
                 learning_rate,
                 exploration_rate,
                 discount_factor,
                 numberOfGames,
                 decay_rate,
                 savingPath=None,
                 observationTransformer=StandardObservationTransformer()
                 ):
        self.onlyOneLife = onlyOneLife
        self.envObsType = envObsType
        self.learningRate = learning_rate
        self.explorationRate = exploration_rate
        self.discountFactor = discount_factor
        self.numberOfGames = numberOfGames
        self.decayRate = decay_rate
        environment = Environment(onlyOneLife, envObsType, observationTransformer)
        q_net = self.init_q_net(environment.env)
        self.deepQLearning = DeepQLearning(environment, q_net, learning_rate, exploration_rate, discount_factor, numberOfGames,
                                           self.decayRate,
                                           self._initSavingPath(savingPath))

    def init_q_net(self, environment):
        init = tf.keras.initializers.HeUniform()
        model = keras.Sequential()

        model.add(keras.layers.Dense(24, input_shape=environment.observation_space.shape, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(len(environment.action_space), activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=self.learningRate),
                      metrics=['accuracy'])
        return model

    def doRun(self):
        self.deepQLearning.deepQLearn()


    @staticmethod
    def _initSavingPath(savingPath):
        if savingPath is None:
            savingPath = root_path(ignore_cwd=False) + '/qTables/'

        if not path.exists(savingPath):
            raise PathError
        else:
            return savingPath
