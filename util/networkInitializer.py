import tensorflow as tf
from tensorflow import keras
from util.options import OPT_GAME_BREAKOUT, OPT_GAME_CARTPOLE


def init_q_net(environment, learningRate):
    if environment.env.spec.id == OPT_GAME_CARTPOLE:
        return init_q_net_cartpole(environment, learningRate)
    else:
        return init_q_net_breakout(environment, learningRate)


def init_q_net_breakout(environment, learningRate):
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()

    model.add(keras.layers.Dense(100, input_shape=environment.env.observation_space.shape, activation='relu',
                                 kernel_initializer=init))
    model.add(keras.layers.Dense(30, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(4, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
                  metrics=['accuracy'])
    return model


def init_q_net_cartpole(environment, learningRate):
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=environment.env.observation_space.shape, activation='relu',
                                 kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(environment.env.action_space.n, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
                  metrics=['accuracy'])
    return model
