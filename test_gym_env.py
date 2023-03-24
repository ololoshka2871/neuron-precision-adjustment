#!/usr/bin/env python

import gym
import gym_quarz
from gym.utils.env_checker import check_env

env = gym.make("gym_quarz/QuartzEnv-v3", render_mode='rgb_array')

check_env(env.unwrapped, skip_render_check=False)