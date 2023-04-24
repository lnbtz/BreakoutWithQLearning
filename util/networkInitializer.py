import tensorflow as tf
from tensorflow import keras


def init_q_net(environment, learningRate):
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()

    model.add(keras.layers.Dense(24, input_shape=environment.env.observation_space.shape, activation='relu',
                                 kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(4, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learningRate),
                  metrics=['accuracy'])
    return model
