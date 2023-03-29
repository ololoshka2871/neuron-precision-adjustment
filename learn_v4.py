#!/usr/bin/env python

import numpy as np

from keras.optimizers import adam_legacy

import tensorflow as tf
import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
from misc.EnvBackCompability import EnvBackCompability

import gym_quarz

from controllers.controller_v4 import NNController

from constants_v4 import *

def learn_main(filename: str,
               steps: int,
               max_episode_steps: int = 200,
               learning_rate=0.001) -> None:

    env = gym.make("gym_quarz/QuartzEnv-v4")
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = EnvBackCompability(env)  # type: ignore

    dqn = NNController(env.observation_space, env.action_space, theta=THETA)

    dqn.compile(adam_legacy.Adam(learning_rate=learning_rate), metrics=['mse'])
    tf.compat.v1.experimental.output_all_intermediates(True)
    dqn.fit(env, nb_steps=steps, visualize=False, verbose=1)

    # After training is done, we save the final weights.
    dqn.save_weights(filename, overwrite=True)

    env.close()


if __name__ == '__main__':
    import argparse

    # parse argumants
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, help='Max iterations', default=10000)
    parser.add_argument('-s', type=float, help='Max steps', default=200)
    parser.add_argument(
        'file', type=str, help='Weigth file', nargs='?', default='learn_v4.h5') # в .tf не сохраняет
    args = parser.parse_args()

    learn_main(args.file, args.i, args.s, learning_rate=0.0005)
