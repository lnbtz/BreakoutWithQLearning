import numpy as np
from environment.observationTransformers.StackedGreyscaleObservationTransformer import StackedGreyscaleObservationTransformer
from environment.Environment import Environment
from util.options import *
import tensorflow as tf
import gymnasium as gym
from matplotlib import pyplot as plt
from keras.models import Model
from deepQLearning.QNet import QNet
from get_project_root import root_path
from keras.preprocessing.image import *
import _pickle as cPickle
import bz2

def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

def show_filter_output(model):
    layers = model.layers

    for layer in layers:
        if 'conv' in layer.name:
            filters, biases = layer.get_weights()
            print(layers[1].name, filters.shape)

            fig1=plt.figure(figsize=(8,12))
            columns = 8
            rows = int(filters.shape[3]/8)
            n_filters = columns * rows
            for i in range(1,n_filters + 1):
                f = filters[:,:,:,i-1]
                fig1 = plt.subplot(rows,columns, i)
                fig1.set_xticks([])
                fig1.set_yticks([])
                plt.imshow(f[:,:,0], cmap='gray')

    plt.show()

def show_feature_maps(model):
    layers = model.layers

    conv_layer_index = [1, 2, 3]
    outputs = [model.layers[i].output for i in conv_layer_index]
    model_short = Model(inputs=model.inputs, outputs=outputs)
    print(model_short.summary())

    env = Environment(OPT_GAME_BREAKOUT, True, OPT_ENV_GREYSCALE, StackedGreyscaleObservationTransformer())
    env.reset()

    observation, _, _, _ = env.step(1)
    observation1, _, _, _ = env.step(3)
    observation2, _, _, _ = env.step(3)
    observation3, _, _, _ = env.step(3)
    observation4, _, _, _ = env.step(3)
    observation5, _, _, _ = env.step(3)
    reshaped_state = observation5 / 255
    reshaped_state = tf.expand_dims(reshaped_state, 0)

    feature_output = model_short.predict(reshaped_state)
    columns = 8
    rows = 8
    for ftr in feature_output:
        fig = plt.figure(figsize=(12,12))
        for i in range(1, len(ftr[0, 0, 0]) + 1):
            fig = plt.subplot(rows, columns, i)
            fig.set_xticks([])
            fig.set_yticks([])
            plt.imshow(ftr[0, :, :, i - 1], cmap='gray')
    plt.show()



if __name__ == "__main__":
    model = QNet.load_model_from_file(root_path(ignore_cwd=True) + "/specialNets/breakoutLeonKindaWorking")
    show_filter_output(model)
    show_feature_maps(model)
