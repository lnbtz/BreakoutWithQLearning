import tensorflow as tf
from matplotlib import pyplot as plt
from keras.models import Model


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

def show_feature_maps(model, observation):
    layers = model.layers

    conv_layer_index = [0, 1, 2, 3]
    outputs = [model.layers[i].output for i in conv_layer_index]
    model_short = Model(inputs=model.inputs, outputs=outputs)
    print(model_short.summary())

#   env = Environment(OPT_GAME_BREAKOUT, True, OPT_ENV_GREYSCALE, StackedGreyscaleObservationTransformer())
#   env.reset()

#   observation, _, _, _ = env.step(1)
#   observation1, _, _, _ = env.step(3)
#   observation2, _, _, _ = env.step(3)
#   observation3, _, _, _ = env.step(3)
#   observation4, _, _, _ = env.step(3)
#   observation5, _, _, _ = env.step(3)
    reshaped_state = observation / 255
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



#if __name__ == "__main__":
    # observations sammeln
#-    qNets_path = root_path(ignore_cwd=True) + '/specialNets/breakoutProgress/'
#-    file_name = "5.52"
    # file_name = "breakoutLeonKindaWorking"

#   qNet = QNet.loadFromFile(qNets_path + file_name)
#   env = Environment(OPT_GAME_BREAKOUT, False, OPT_ENV_GREYSCALE, StackedGreyscaleObservationTransformer())

#   observations = showQGame(env, qNet)


#-    model = QNet.load_model_from_file(root_path(ignore_cwd=True) + "/specialNets/breakoutProgress/73.17")
#-    show_filter_output(model)

#   obs1 = observations[18]
#   obs2 = observations[360]
#   show_feature_maps(model, obs1)
#   show_feature_maps(model, obs2)