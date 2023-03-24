#!/usr/bin/env python

import gym
import gym_quarz
from gym.utils.env_checker import check_env

env = gym.make("gym_quarz/QuartzEnv-v3", render_mode='human')

env.reset()

for _ in range(1000):
    s, r, done, truncated, info = env.step(env.action_space.sample()) # take a random action
    if not done:
        env.render()

env.close()