#!/usr/bin/env python


import os
import pickle

import numpy as np

from keras.optimizers import adam_v2

import gym
import gym_quarz

from controllers.controller_v4 import NNController


def learn_main(steps: int,
               learning_rate=0.001) -> None:

    env = gym.make("gym_quarz/QuartzEnv-v3")
    actions: int = env.action_space.shape[0]  # type: ignore
    states: int = env.observation_space.shape[0] # type: ignore
    
    dqn = NNController(states, actions)

    dqn.compile(adam_v2.Adam(learning_rate=learning_rate), metrics=['mse'])
    dqn.fit(env, nb_steps=steps, visualize=True, verbose=2)

    scores = dqn.test(env, nb_episodes=3, visualize=True)
    print(np.mean(scores.history['episode_reward']))

    env.close()


if __name__ == '__main__':
    import argparse

    # parse argumants
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, help='Max iterations', default=0)
    parser.add_argument(
        'file', type=str, help='Simulation history file', nargs='?', default='learn_v4.ckl')
    args = parser.parse_args()

    learn_main(args.m,
               learning_rate=0.0005)
