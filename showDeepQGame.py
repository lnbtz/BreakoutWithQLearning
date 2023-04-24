from QNet import QNet
import pygame
import gymnasium as gym
from get_project_root import root_path

PLAY_SPEED = 50
DELAY_AFTER_DEATH = 2  # Seconds

qNets_path = root_path(ignore_cwd=False) + '/qNets/'
file_name = "model"

game_name = 'CartPole-v1'

qNet = QNet(qNets_path + "model")
env = gym.make(game_name, render_mode="rgb_array")

# Initialize pygame
observation, info = env.reset(seed=5)

ary = env.render()
width = len(ary[0])
height = len(ary)

pygame.init()
display = pygame.display.set_mode((width, height))
pygame.display.set_caption(game_name)
clock = pygame.time.Clock()

total_rewards = 0
running = True
while running:
    action = qNet.getBestAction(observation)
    observation, reward, terminated, _, info = env.step(action)
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
