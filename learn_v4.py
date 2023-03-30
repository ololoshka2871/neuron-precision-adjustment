#!/usr/bin/env python

import numpy as np

from keras.optimizers import adam_legacy

from rl.callbacks import Callback

import tensorflow as tf
import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
from misc.EnvBackCompability import EnvBackCompability

import gym_quarz

from controllers.controller_v4 import NNController

from constants_v4 import *


class AbortCatcher(Callback):
    def __init__(self):
        super(AbortCatcher, self).__init__()
        self.aborted = False

    def on_train_end(self, logs={}):
        if logs['did_abort']:
            print(">>> Training aborted")
            self.aborted = True


def learn_main(filename: str,
               initial_file: str,
               steps: int,
               time_limit: float = 10.0,
               checkpoint_every: int = 100,
               learning_rate=0.001) -> None:

    env = gym.make("gym_quarz/QuartzEnv-v4", time_limit=time_limit)
    env = EnvBackCompability(env)  # type: ignore

    dqn = NNController(env.observation_space, env.action_space,
                       theta=THETA, batch_size=BATCH_SIZE)

    dqn.compile(adam_legacy.Adam(learning_rate=learning_rate), metrics=['mse'])
    tf.compat.v1.experimental.output_all_intermediates(True)

    if initial_file:
        print(f">>> Load initial weights from {initial_file}")
        dqn.load_weights(initial_file)

    for i in range(steps // checkpoint_every):
        ac = AbortCatcher()
        dqn.fit(env, nb_steps=checkpoint_every, callbacks=[ac], visualize=False, verbose=1)
        if ac.aborted:
            return
        dqn.save_weights(f"cp{i:04}-{filename}", overwrite=True)
        print(f">>> Checkpoint cp{i:04}-{filename} saved")
    
    if steps % checkpoint_every != 0:
        ac = AbortCatcher()
        dqn.fit(env, nb_steps=steps - (steps // checkpoint_every)
                * checkpoint_every, callbacks=[ac], visualize=False, verbose=1)
        if ac.aborted:
            return
        dqn.save_weights(f"end-{filename}", overwrite=True)

    env.close()


if __name__ == '__main__':
    import argparse

    # parse argumants
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', type=float, help='Time limit (s.)', default=10.0)
    parser.add_argument('-i', type=int, help='Max iterations', default=10000)
    parser.add_argument('-c', type=int, help='Сheckpoint evry', default=1000)
    parser.add_argument('-s', type=str, help='Initial weights file .h5')
    parser.add_argument(
        'file', type=str, help='Weigth file base (chX-filename.h5)', nargs='?', default='learn_v4.h5')  # в .tf не сохраняет
    args = parser.parse_args()

    learn_main(args.file, args.s, args.i, args.l, args.c, learning_rate=0.0005)
