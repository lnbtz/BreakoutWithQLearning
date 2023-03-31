import gymnasium as gym
from observationTransformers.StandardObservationTransformer import StandardObservationTransformer


class Environment:
    GAME = "BreakoutDeterministic-v4"

    def __init__(self, onlyOneLife, envObsType,
                 observationTransformer=StandardObservationTransformer()):
        self.onlyOneLife = onlyOneLife
        self.observationTransformer = observationTransformer
        self.env = gym.make(self.GAME, render_mode="rgb_array", obs_type=envObsType)
        self.observationSpaceSize = self.env.observation_space.n
        self.actionSpace = self.env.action_space.n
        self.env.reset()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        if self.onlyOneLife and info['lives'] < 5:
            terminated = True

        return self.observationTransformer.transform(observation), reward, terminated

    def reset(self):
        self.env.reset()

    def close(self):
        self.env.close()
