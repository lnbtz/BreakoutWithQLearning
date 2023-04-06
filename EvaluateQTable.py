from Environment import Environment


def evaluateQTable(qtable):
    only_one_life, env_obs_type, _, _, _, observation_transformer = qtable.trainingParameter
    env = Environment(only_one_life, env_obs_type, observation_transformer)
    observation, info = env.reset()

    score = 0
    game_over = False
    while not game_over:
        action = qtable.getBestAction(observation)
        observation, reward, terminated = env.step(action)

        score = score + reward

        if terminated:
            game_over = True

    return score
