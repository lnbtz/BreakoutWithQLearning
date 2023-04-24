import gymnasium as gym
from gymnasium.utils.play import play

env = gym.make("BreakoutDeterministic-v4", render_mode="rgb_array")

play(env, keys_to_action={
    "a" : 3,
    "d" : 2,
    "w" : 1
})
