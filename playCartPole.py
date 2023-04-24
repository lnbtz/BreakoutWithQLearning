import gymnasium as gym
from gymnasium.utils.play import play

env = gym.make('CartPole-v1', render_mode="rgb_array")
play(env, keys_to_action={
    "a" : 0,
    "d" : 1
})

