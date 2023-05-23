import os.path

from deepQLearning.DeepQLearning import DeepQLearning
from environment.Environment import Environment
from environment.observationTransformers.StandardObservationTransformer import StandardObservationTransformer
from get_project_root import root_path
from util.networkInitializer import init_q_net_breakout, init_q_net_cartpole
from util.options import OPT_GAME_BREAKOUT, OPT_GAME_CARTPOLE


class WrongGameError(Exception):
    pass


class Config:
    def __init__(self,
                 game,
                 only_one_life,
                 env_obs_type,
                 learning_rate,
                 exploration_rate,
                 min_exploration_rate,
                 discount_factor,
                 solution_running_reward,
                 decay_rate,
                 backpropagation_rate,
                 replay_memory_length,
                 batch_size,
                 copy_step_limit,
                 max_exploration_rate,
                 exploration_frames,
                 max_steps_per_episode,
                 epsilon_greedy_frames,
                 saving_path=None,
                 observation_transformer=StandardObservationTransformer()
                 ):
        environment = Environment(game, only_one_life, env_obs_type, observation_transformer)

        if game == OPT_GAME_BREAKOUT:
            q_net = init_q_net_breakout(environment.env, learning_rate)
        elif game == OPT_GAME_CARTPOLE:
            q_net = init_q_net_cartpole(environment.env, learning_rate)
        else:
            raise WrongGameError

        self.deepQLearning = DeepQLearning(environment,
                                           q_net,
                                           learning_rate,
                                           exploration_rate,
                                           min_exploration_rate,
                                           discount_factor,
                                           solution_running_reward,
                                           decay_rate,
                                           backpropagation_rate,
                                           replay_memory_length,
                                           batch_size,
                                           copy_step_limit,
                                           max_exploration_rate,
                                           exploration_frames,
                                           max_steps_per_episode,
                                           epsilon_greedy_frames,
                                           self._init_saving_path(saving_path)
                                           )

    def do_run(self):
        self.deepQLearning.deepQLearn()

    @staticmethod
    def _init_saving_path(saving_path):
        if saving_path is None:
            return os.path.join(root_path(ignore_cwd=False), "qNets", "default")
        else:
            return saving_path
