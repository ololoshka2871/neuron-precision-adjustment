#!/usr/bin/env python

from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium.utils.play import play

import pygame

import gym_quarz

from misc.MyPlayPlot_v5 import MyPlayPlot_v5


plotter = MyPlayPlot_v5(30 * 5)

env = gym.make("gym_quarz/QuartzEnv-v6", render_mode='rgb_array')

env.reset()

frame = env.render()
video_size = frame.shape[:2]  # type: ignore
screen = pygame.display.set_mode(video_size)
fps = env.metadata.get("render_fps", 60)

mapping = {(pygame.K_ESCAPE,): 0, 
           (pygame.K_DOWN,): 1,
           (pygame.K_UP,): 2, 
           (pygame.K_SPACE,): 3,
           (pygame.K_RIGHT,): 4,
           (pygame.K_LEFT,): 5,
           (pygame.K_END,): 7,}
play(env, callback=plotter.callback,
     keys_to_action=mapping, noop=0)  # type: ignore
