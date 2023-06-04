import tensorflow as tf
from keras_visualizer import visualizer
from tensorflow import keras
from util.options import OPT_GAME_BREAKOUT, OPT_GAME_CARTPOLE
from keras import layers

def init_q_net(environment, learningRate):
    if environment.env.spec.id == OPT_GAME_CARTPOLE:
        return init_q_net_cartpole(environment, learningRate)
    else:
        return init_q_net_breakout(environment, learningRate)


def init_q_net_breakout(environment, learningRate):
    # init = tf.keras.initializers.HeUniform()
    # model = keras.Sequential()
    #
    # model.add(keras.layers.Dense(100, input_shape=environment.env.observation_space.shape, activation='relu',
    #                              kernel_initializer=init))
    # model.add(keras.layers.Dense(30, activation='relu', kernel_initializer=init))
    # model.add(keras.layers.Dense(4, activation='linear', kernel_initializer=init))
    # model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
    #               metrics=['accuracy'])
    # return model
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(84, 84, 4,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(4, activation="linear")(layer5)

    model = keras.Model(inputs=inputs, outputs=action)
    visualizer(model, file_format='png', view=True)
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
