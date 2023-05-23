import gymnasium as gym
from util.options import OPT_GAME_BREAKOUT, OPT_GAME_CARTPOLE


class Environment:

    def __init__(self,
                 game,
                 onlyOneLife,
                 envObsType,
                 observationTransformer):
        self.onlyOneLife = onlyOneLife
        self.observationTransformer = observationTransformer
        self.game = game
        if game == OPT_GAME_BREAKOUT:
            self.env = gym.make(self.game, render_mode="rgb_array", obs_type=envObsType)
        else:
            self.env = gym.make(self.game, render_mode="rgb_array")
        self.lives = 5

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.onlyOneLife and info['lives'] < self.lives:
            self.lives = info['lives']
            terminated = True
        return self.observationTransformer.transform(observation), reward, terminated

    def reset(self):
        obs, _ = self.env.reset()
        return self.observationTransformer.transform(obs)

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()
