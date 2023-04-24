from environment.Environment import Environment
from qLearning.QTable import QTable
import pygame
from environment.observationTransformers.StandardObservationTransformer import StandardObservationTransformer
from options import *
from get_project_root import root_path

PLAY_SPEED = 50

qTables_path = root_path(ignore_cwd=False) + '/qTables/'
file_name = '04|15|2023|19:03:28.qtable'
path = qTables_path + file_name

env = Environment(True, OPT_ENV_RAM, StandardObservationTransformer())
q_table = QTable(env.actionSpaceSize, path)


# Initialize pygame
observation, info = env.reset()

ary = env.render()
width = len(ary[0])
height = len(ary)

pygame.init()
display = pygame.display.set_mode((width, height))
pygame.display.set_caption('Breakout')
clock = pygame.time.Clock()

running = True
while running:
    action = q_table.getBestAction(observation)
    observation, reward, terminated = env.step(action)

    ary = env.render()
    img = pygame.surfarray.make_surface(ary)
    img = pygame.transform.rotate(img, -90)
    img = pygame.transform.flip(img, True, False)
    display.blit(img, (0, 0))
    pygame.display.update()
    clock.tick(PLAY_SPEED)

    if terminated:
        observation, info = env.reset()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            running = False
