#!/usr/bin/env python

import numpy as np

from keras.optimizers import adam_legacy

import tensorflow as tf
import gymnasium as gym
from misc.EnvBackCompability import EnvBackCompability

import gym_quarz

from controllers.controller_v4 import NNController

from constants_v4 import *


def display_main(filename: str,
                 time_limit: float,
                 episodes: int) -> None:
    env = gym.make("gym_quarz/QuartzEnv-v4", render_mode='human', time_limit=time_limit)
    env = EnvBackCompability(env)  # type: ignore

    dqn = NNController(env.observation_space, env.action_space, theta=THETA, batch_size=BATCH_SIZE)
    
    # Сначала надо скомпилировать модель, иначе не загрузится веса
    dqn.compile(adam_legacy.Adam(), metrics=['mse'])
    tf.compat.v1.experimental.output_all_intermediates(True)
    
    if filename != 'none':
        dqn.load_weights(filename)
    
    # freeze model
    dqn.training = False

    scores = dqn.test(env, nb_episodes=episodes, visualize=True)

    env.close()


if __name__ == '__main__':
    import argparse

    # parse argumants
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=float, help='Time limit (s.)', default=10.0)
    parser.add_argument('-e', type=int, help='Episodes', default=1)
    parser.add_argument(
        'file', type=str, help='Weigth file', nargs='?', default='learn_v4.h5')  # в .tf не сохраняет
    args = parser.parse_args()

    display_main(args.file, args.l, args.e)
