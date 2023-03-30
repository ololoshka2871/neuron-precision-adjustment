#!/usr/bin/env python

import gymnasium as gym
import gym_quarz
from gymnasium.utils.env_checker import check_env

env = gym.make("gym_quarz/QuartzEnv-v4", render_mode='rgb_array')

check_env(env.unwrapped, skip_render_check=False)