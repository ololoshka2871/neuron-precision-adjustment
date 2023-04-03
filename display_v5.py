#!/usr/bin/env python

from keras.optimizers import adam_legacy

import gymnasium as gym
import numpy as np

from rl.callbacks import Callback

from misc.EnvBackCompability import EnvBackCompability

import gym_quarz

from controllers.controller_v5 import DQNController as Controller
from misc.MyPlayPlot_v5 import MyPlayPlot_v5

from constants_v5 import *


class MyCallback(Callback):
    def __init__(self, points=150):
        super(MyCallback, self).__init__()
        self.plot = MyPlayPlot_v5(points)

    def on_episode_begin(self, episode, logs):
        for data_series in self.plot.data:
            data_series.clear()
            self.plot.t = 0

    def on_step_end(self, step, logs={}):
        """Called at end of each step"""
        obs = logs['observation']
        non_mod_obs = np.array([obs[2], obs[3], 1.0 / obs[0], obs[1]])
        self.plot.callback(
            None,
            non_mod_obs,
            logs['action'],
            logs['reward'],
            False, False,
            logs['info'])

    def on_episode_end(self, episode, logs):
        info = logs['info']
        precision = info['static_freq_change'] / info['adjust_target'] * 100
        print(
            f"""
- Adjustment {info['static_freq_change']:.2f} Hz/{info['adjust_target']:.2f} Hz: {precision:.2f}%,
- Penalty energy: {info['penalty_energy']},
- dissbalance: {info['disbalance'] * 100:.2f} %,
- Time: {info['time_elapsed']:.2f} s.
""")


def display_main(filename: str,
                 time_limit: float,
                 episodes: int) -> None:
    env = gym.make("gym_quarz/QuartzEnv-v5", render_mode='human',
                   time_limit=time_limit)
    env = EnvBackCompability(env)  # type: ignore

    dqn = Controller(env.observation_space, env.action_space)

    # Сначала надо скомпилировать модель, иначе не загрузится веса
    dqn.compile(adam_legacy.Adam(), metrics=['mae'])

    if filename != 'none':
        dqn.load_weights(filename)

    # freeze model
    dqn.training = False

    dqn.test(env, callbacks=[MyCallback()],
             nb_episodes=episodes, visualize=True)

    env.close()


if __name__ == '__main__':
    import argparse

    # parse argumants
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=float, help='Time limit (s.)', default=10.0)
    parser.add_argument('-e', type=int, help='Episodes', default=1)
    parser.add_argument(
        'file', type=str, help='Weights file (none = random)', nargs='?', default='none')  # в .tf не сохраняет
    args = parser.parse_args()

    display_main(args.file, args.l, args.e)
