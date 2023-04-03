#!/usr/bin/env python

from typing import Optional

from keras.optimizers import adam_legacy

from rl.callbacks import ModelIntervalCheckpoint, FileLogger

import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
from misc.EnvBackCompability import EnvBackCompability

import gym_quarz

from controllers.controller_v5 import DQNController as Controller

from constants_v5 import *


def learn_main(filename: str,
               initial_file: str,
               steps: int,
               time_limit: float = 10.0,
               checkpoint_every: int = 100,
               log_file_name: Optional[str] = None,
               learning_rate=0.001) -> None:

    env = gym.make("gym_quarz/QuartzEnv-v5",
                   time_limit=time_limit)
    env = EnvBackCompability(env)  # type: ignore

    dqn = Controller(env.observation_space, 
                     env.action_space)

    dqn.compile(adam_legacy.Adam(learning_rate=learning_rate), metrics=['mae'])

    if initial_file:
        print(f">>> Load initial weights from {initial_file}")
        dqn.load_weights(initial_file)

    cp_saver = ModelIntervalCheckpoint(
        f"step{{step:08}}-{filename}", interval=checkpoint_every)
    callbacks: list[object] = [cp_saver]
    if log_file_name is not None:
        callbacks.append(FileLogger(log_file_name, interval=checkpoint_every))

    dqn.fit(env, nb_steps=steps, callbacks=callbacks, visualize=False, verbose=1)

    if steps % checkpoint_every != 0:
        dqn.save_weights(f"end-{filename}", overwrite=True)

    env.close()


if __name__ == '__main__':
    import argparse

    # parse argumants
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=float, help='Time limit (s.)', default=10.0)
    parser.add_argument('-i', type=int, help='Max iterations', default=10000)
    parser.add_argument(
        '-c', type=int, help='Сheckpoint every c steps', default=1000)
    parser.add_argument('-s', type=str, help='Initial weights file .h5')
    parser.add_argument(
        'file', type=str, help='Weigth file base (stepX-filename.h5)', nargs='?', default='learn_v5.h5')
    parser.add_argument('--log', type=str, help='log file name (default: learn_v5.ljson)', default='learn_v5.ljson')
    args = parser.parse_args()

    learn_main(args.file, args.s, args.i, args.l, args.c, args.log, learning_rate=0.0005)
