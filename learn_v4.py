#!/usr/bin/env python


import numpy as np

from keras.optimizers import adam_legacy  # keras 2.12.0

import tensorflow as tf
import gymnasium as gym
from misc.EnvBackCompability import EnvBackCompability

import gym_quarz

from controllers.controller_v4 import NNController


def learn_main(filename: str,
               steps: int,
               learning_rate=0.001) -> None:

    env = gym.make("gym_quarz/QuartzEnv-v3")
    env = EnvBackCompability(env)  # type: ignore

    dqn = NNController(env.observation_space, env.action_space)

    dqn.compile(adam_legacy.Adam(learning_rate=learning_rate), metrics=['mse'])
    tf.compat.v1.experimental.output_all_intermediates(True)
    dqn.fit(env, nb_steps=steps, visualize=False, verbose=1)

    # After training is done, we save the final weights.
    dqn.save_weights(filename, overwrite=True)

    env.close()

    # ------------------------------------

    #display_env = gym.make("gym_quarz/QuartzEnv-v3", render_mode='human')
    #display_env = EnvBackCompability(display_env)  # type: ignore
    #scores = dqn.test(display_env, nb_episodes=3, visualize=True)
    #print(np.mean(scores.history['episode_reward']))
    #
    #display_env.close()


if __name__ == '__main__':
    import argparse

    # parse argumants
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, help='Max iterations', default=1000)
    parser.add_argument(
        'file', type=str, help='Simulation history file', nargs='?', default='learn_v4.h5f')
    args = parser.parse_args()

    learn_main(args.file,
               args.m,
               learning_rate=0.0005)
