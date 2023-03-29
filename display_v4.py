#!/usr/bin/env python

import numpy as np

from keras.optimizers import adam_legacy

import tensorflow as tf
import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
from misc.EnvBackCompability import EnvBackCompability

import gym_quarz

from controllers.controller_v4 import NNController


def display_main(filename: str,
                 max_episode_steps: int) -> None:
    env = gym.make("gym_quarz/QuartzEnv-v4", render_mode='human')
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = EnvBackCompability(env)  # type: ignore

    dqn = NNController(env.observation_space, env.action_space)
    
    # Сначала надо скомпилировать модель, иначе не загрузится веса
    dqn.compile(adam_legacy.Adam(learning_rate=0.001), metrics=['mse'])
    tf.compat.v1.experimental.output_all_intermediates(True)
    
    if filename != 'none':
        dqn.load_weights(filename)
    
    # freeze model
    dqn.training = False

    scores = dqn.test(env, nb_episodes=1, visualize=True)
    print(np.mean(scores.history['episode_reward']))

    env.close()


if __name__ == '__main__':
    import argparse

    # parse argumants
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=float, help='Max steps', default=200)
    parser.add_argument(
        'file', type=str, help='Weigth file', nargs='?', default='learn_v4.h5')  # в .tf не сохраняет
    args = parser.parse_args()

    display_main(args.file,
                 args.s)
