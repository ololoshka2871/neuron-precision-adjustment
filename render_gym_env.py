#!/usr/bin/env python

from typing import List

import numpy as np

import gymnasium as gym
from gymnasium.utils.play import play, PlayPlot, display_arr

import pygame

import gym_quarz

def callback(obs_t, obs_tp1, action, reward, done, truncated, info):
    return [reward, obs_tp1[4], info['static_freq_change'], info['penalty_energy'], info['temperature']]


plotter = PlayPlot(callback, 30 * 5, 
    ["Reward", "Current offset", "Static freq change", "Penalty energy", "Temperature"]
)

env = gym.make("gym_quarz/QuartzEnv-v4", render_mode='rgb_array')

env.reset()

frame = env.render()
video_size = frame.shape[:2]  # type: ignore
screen = pygame.display.set_mode(video_size)
fps = env.metadata.get("render_fps", 30)

clock = pygame.time.Clock()
done, obs = False, None
total_reward = 0.0
info = dict()
while not done:
    action = env.action_space.sample()
    prev_obs = obs
    obs, rew, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += rew  # type: ignore
    plotter.callback(prev_obs, obs, action, rew, terminated, truncated, info)
    if obs is not None:
        rendered = env.render()
        if isinstance(rendered, List):
            rendered = rendered[-1]
        assert rendered is not None and isinstance(rendered, np.ndarray)
        display_arr(screen, rendered, transpose=True, video_size=video_size)  # type: ignore

    pygame.display.flip()
    clock.tick(fps)
    
pygame.quit()

print("Done!")
print(f"Total adjust: {info['static_freq_change']}, total reward: {total_reward}")