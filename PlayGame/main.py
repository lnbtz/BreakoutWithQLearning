import gymnasium
import numpy as np
import h5py
import time


env = gymnasium.make("Taxi-v3", render_mode="human")


# play game with q table
with h5py.File("q_table.h5", "r") as f:
    q_table = f["q_table"][:]

state = env.reset()[0]
done = False

while not done:
    action = np.argmax(q_table[state])
    time.sleep(0.5)
    state, reward, done, truncated, info = env.step(action)
