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
            self.live_counter = 5
        else:
            self.env = gym.make(self.game, render_mode="rgb_array")
        self.env.reset(seed=42)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        ball_dropped = False

        if self.onlyOneLife and info['lives'] < 5:
            terminated = True
            ball_dropped = True
        else:
            if info['lives'] < self.live_counter:
                ball_dropped = True
                self.live_counter = info['lives']
            elif info['lives'] > self.live_counter:
                self.live_counter = info['lives']

        return self.observationTransformer.transform(observation), reward, terminated, ball_dropped


    def reset(self):
        obs, info = self.env.reset(seed=42)
        return self.observationTransformer.transform(obs)

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()
