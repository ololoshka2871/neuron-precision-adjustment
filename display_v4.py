#!/usr/bin/env python

import timeit
import warnings

import numpy as np

from keras.optimizers import adam_legacy

import tensorflow as tf
import gymnasium as gym
from gymnasium.utils.play import PlayPlot

from rl.callbacks import Callback

from misc.EnvBackCompability import EnvBackCompability

import gym_quarz

from controllers.controller_v4 import NNController

from constants_v4 import *


class MyCallback(Callback):
    @staticmethod
    def filter(obs_t, obs_tp1, action, reward, done, truncated, info):
        return [reward, obs_tp1[5], info['static_freq_change'], info['penalty_energy'], info['temperature']]

    def __init__(self, points=150):
        super(MyCallback, self).__init__()
        self.plot = PlayPlot(MyCallback.filter, points,
                             ["Reward", "Current offset", "Static freq change", "Penalty energy", "Temperature"])

    def on_episode_begin(self, episode, logs):
        for data_series in self.plot.data:
            data_series.clear()

        return super().on_episode_begin(episode, logs)

    def on_step_end(self, step, logs={}):
        """Called at end of each step"""
        self.plot.callback(
            None,
            logs['observation'],
            logs['action'],
            logs['reward'],
            False, False,
            logs['info'])


def display_main(filename: str,
                 time_limit: float,
                 episodes: int) -> None:
    env = gym.make("gym_quarz/QuartzEnv-v4", render_mode='human',
                   time_limit=time_limit, relative_move=RELATIVE)
    env = EnvBackCompability(env)  # type: ignore

    dqn = NNController(env.observation_space, env.action_space,
                       theta=THETA, batch_size=BATCH_SIZE)

    # Сначала надо скомпилировать модель, иначе не загрузится веса
    dqn.compile(adam_legacy.Adam(), metrics=['mse'])
    tf.compat.v1.experimental.output_all_intermediates(True)

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
