from deepQLearning.QNet import QNet
import pygame
from environment.Environment import Environment
from environment.observationTransformers.StandardObservationTransformer import StandardObservationTransformer
from options import *
from get_project_root import root_path

PLAY_SPEED = 50
DELAY_AFTER_DEATH = 2  # Seconds

qNets_path = root_path(ignore_cwd=True) + '/qNets/'
file_name = "test"

qNet = QNet.loadFromFile(qNets_path + file_name)
env = Environment(True, OPT_ENV_RAM, StandardObservationTransformer())

# Initialize pygame
observation, info = env.reset()

ary = env.render()
width = len(ary[0])
height = len(ary)

pygame.init()
display = pygame.display.set_mode((width, height))
pygame.display.set_caption('Breakout - Q-Agent')
clock = pygame.time.Clock()

total_rewards = 0
running = True
while running:
    action = qNet.getBestAction(observation)
    observation, reward, terminated = env.step(action)
    total_rewards += reward

    ary = env.render()
    img = pygame.surfarray.make_surface(ary)
    img = pygame.transform.rotate(img, -90)
    img = pygame.transform.flip(img, True, False)
    display.blit(img, (0, 0))
    pygame.display.update()
    clock.tick(PLAY_SPEED)

    if terminated:
        observation, info = env.reset()
        print("Total Reward: " + str(total_rewards))
        total_rewards = 0
        pygame.time.wait(DELAY_AFTER_DEATH * 1000)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            running = False
