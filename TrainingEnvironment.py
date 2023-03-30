import gymnasium as gym
import pygame
from observationTransformers.StandardObservationTransformer import StandardObservationTransformer


class TrainingEnvironment:
    GAME = "BreakoutDeterministic-v4"
    PLAY_SPEED = 50

    def __init__(self, q, onlyOneLife, renderTraining, envObsType,
                 observationTransformer=StandardObservationTransformer()):
        self.q = q
        self.onlyOneLife = onlyOneLife
        self.renderTraining = renderTraining
        self.observationTransformer = observationTransformer
        self.env = gym.make(self.GAME, render_mode="rgb_array", obs_type=envObsType)

    def doTraining(self, numberOfRuns):
        if self.renderTraining:
            self._doTrainingWithRendering(numberOfRuns)
        else:
            self._doTrainingWithoutRendering(numberOfRuns)

    def _doTrainingWithRendering(self, numberOfRuns):
        observation, info = self.env.reset()

        ary = self.env.render()
        width = len(ary[0])
        height = len(ary)

        pygame.init()
        display = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Breakout')
        clock = pygame.time.Clock()

        running = True
        for _ in range(numberOfRuns):
            if not running:
                break

            action = self.q.getAction(self.observationTransformer.transform(observation))

            observation, reward, terminated, truncated, info = self.env.step(action)

            ary = self.env.render()
            img = pygame.surfarray.make_surface(ary)
            img = pygame.transform.rotate(img, -90)
            img = pygame.transform.flip(img, True, False)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            display.blit(img, (0, 0))
            pygame.display.update()
            clock.tick(self.PLAY_SPEED)

            self.q.updateQValues(reward)

            if self.onlyOneLife and info['lives'] < 5:
                terminated = True

            if terminated or truncated:
                observation, info = self.env.reset()
                self.q.endEpisode()

        self.env.close()
        self.q.endEpisode()
        pygame.quit()

    def _doTrainingWithoutRendering(self, numberOfRuns):
        observation, info = self.env.reset()

        for _ in range(numberOfRuns):
            action = self.q.getAction(self.observationTransformer.transform(observation))

            observation, reward, terminated, truncated, info = self.env.step(action)

            self.q.updateQValues(reward)

            if self.onlyOneLife and info['lives'] < 5:
                terminated = True

            if terminated or truncated:
                observation, info = self.env.reset()
                self.q.endEpisode()

        self.env.close()
        self.q.endEpisode()
