#!/usr/bin/env python

import numpy as np

from keras.optimizers import adam_legacy

from rl.callbacks import ModelIntervalCheckpoint

import tensorflow as tf
import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
from misc.EnvBackCompability import EnvBackCompability

import gym_quarz

from controllers.controller_v4 import NNController

from constants_v4 import *


def learn_main(filename: str,
               initial_file: str,
               steps: int,
               time_limit: float = 10.0,
               checkpoint_every: int = 100,
               learning_rate=0.001) -> None:

    env = gym.make("gym_quarz/QuartzEnv-v4", 
                   time_limit=time_limit,
                   relative_move=RELATIVE)
    env = EnvBackCompability(env)  # type: ignore

    dqn = NNController(env.observation_space, env.action_space,
                       theta=THETA, batch_size=BATCH_SIZE)

    dqn.compile(adam_legacy.Adam(learning_rate=learning_rate), metrics=['mse'])
    tf.compat.v1.experimental.output_all_intermediates(True)

    if initial_file:
        print(f">>> Load initial weights from {initial_file}")
        dqn.load_weights(initial_file)

    cp_saver = ModelIntervalCheckpoint(f"step{{step:08}}-{filename}", interval=checkpoint_every)
    dqn.fit(env, nb_steps=steps, callbacks=[cp_saver], visualize=False, verbose=1)

    if steps % checkpoint_every != 0:
        dqn.save_weights(f"end-{filename}", overwrite=True)

    env.close()


if __name__ == '__main__':
    import argparse

    # parse argumants
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=float, help='Time limit (s.)', default=10.0)
    parser.add_argument('-i', type=int, help='Max iterations', default=10000)
    parser.add_argument('-c', type=int, help='Сheckpoint every c steps', default=1000)
    parser.add_argument('-s', type=str, help='Initial weights file .h5')
    parser.add_argument(
        'file', type=str, help='Weigth file base (stepX-filename.h5)', nargs='?', default='learn_v4.h5')  # в .tf не сохраняет
    args = parser.parse_args()

    learn_main(args.file, args.s, args.i, args.l, args.c, learning_rate=0.0005)
