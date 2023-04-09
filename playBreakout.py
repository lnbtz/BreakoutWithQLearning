
import gymnasium as gym

env = gym.make("BipedalWalker-v3", render_mode="human")

while True:
    env.reset()
    for i in range(1000):
        observation, reward, truncated, done, info = env.step(env.action_space.sample())
        print(observation)
        print("reward: ", reward)
        env.render()

