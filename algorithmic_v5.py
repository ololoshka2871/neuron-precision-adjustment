#!/usr/bin/env python

from copy import deepcopy
import math
from typing import List, Optional, Tuple

import gymnasium as gym
from gymnasium.utils.play import display_arr
import numpy as np
import pygame

import gym_quarz

from controllers.controller_v5_algo import AlgorithmicController as Controller
from misc.MyPlayPlot_v5 import MyPlayPlot_v5


def _get_video_size(env, zoom: Optional[float] = None) -> Tuple[int, int]:
    rendered = env.render()
    if isinstance(rendered, List):
        rendered = rendered[-1]
    assert rendered is not None and isinstance(rendered, np.ndarray)
    video_size = (rendered.shape[1], rendered.shape[0])

    if zoom is not None:
        video_size = (int(video_size[0] * zoom), int(video_size[1] * zoom))

    return video_size


def display_main() -> None:
    env = gym.make("gym_quarz/QuartzEnv-v5", render_mode='rgb_array')

    ctrl = Controller(env.observation_space, env.action_space)
    plotter = MyPlayPlot_v5(30 * 5)

    obs, info = env.reset()

    video_size = _get_video_size(env, zoom=None)
    screen = pygame.display.set_mode(video_size, pygame.RESIZABLE)

    fps = env.metadata.get("render_fps", 30)

    prev_obs, done = deepcopy(obs), False
    clock = pygame.time.Clock()
    while not done:
        action = ctrl.sample_action(prev_obs, obs)
        prev_obs = obs
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        plotter.callback(None, obs, action, rew, terminated, truncated, info)
        if obs is not None:
            rendered = env.render()
            if isinstance(rendered, List):
                rendered = rendered[-1]
            assert rendered is not None and isinstance(rendered, np.ndarray)
            display_arr(screen, rendered, # type: ignore
                        transpose=True, video_size=video_size)

        # process pygame events
        _ = pygame.event.get()

        pygame.display.flip()
        clock.tick(fps)

    # TODO

    env.close()


if __name__ == '__main__':
    display_main()
