import numpy as np
import random
from Config import config
import h5py


def q_learn(environment):
    # build table
    q_table = np.zeros([environment.observation_space.n, environment.action_space.n])

    # start training
    for i in range(1, config.NUMBER_OF_GAMES):
        # setup game
        state = environment.reset()[0]
        epochs = 0
        done = False

        # play game
        while not done:

            if random.uniform(0, 1) < config.EPSILON:
                action = environment.action_space.sample()  # explore
            else:
                action = np.argmax(q_table[state])  # don't explore and take best known move

            # get next state
            next_state, reward, done, truncated, _ = environment.step(action)
            # get last value

            old_value = q_table[state, action]
            # get next best value
            next_max = np.max(q_table[next_state])
            # get new value
            new_value = (1 - config.ALPHA) * old_value + config.ALPHA * (reward + config.GAMMA * next_max)
            # update q table
            q_table[state, action] = new_value
            # update state
            state = next_state

            epochs += 1

        print("done after {} epochs\n".format(epochs))

    # save q table
    with h5py.File("q_table.h5", "w") as f:
        f.create_dataset("q_table", data=q_table)

    print("done training")
