#!/usr/bin/env python

from collections import deque
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import gymnasium as gym
from gymnasium.utils.play import play, display_arr
from gymnasium.core import ActType, ObsType

import pygame

import gym_quarz


class PlayPlot:
    def __init__(self, horizon_timesteps: int):
        self.horizon_timesteps = horizon_timesteps

        titles = ["Reward", "Offset", "Static freq change",
                  "Penalty energy", "Temperature"]
        num_plots = len(titles)

        self.fig, self.ax = plt.subplots(num_plots)
        for axis, name in zip(self.ax, titles):
            axis.set_title(name)

        self.t = 0
        self.cur_plot: List[list[Line2D]] = [
            self.ax[0].plot([], color='green'),  # reward
            # offset + target offset
            self.ax[1].plot([], [], 'b-', [], [], 'r:'),
            # static freq change + target offset
            self.ax[2].plot([], [], 'b-', [], [], 'r:'),
            self.ax[3].plot([], color='red'),  # penalty energy
            self.ax[4].plot([], color='red'),  # temperature
        ]
        self.data = [deque(maxlen=horizon_timesteps) for _ in range(6)]

        for ax in self.ax:
            ax.autoscale(enable=True)

    def callback(
        self,
        obs_t: ObsType,
        obs_tp1: ObsType,
        action: ActType,
        rew: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ):
        """The callback that calls the provided data callback and adds the data to the plots.

        Args:
            obs_t: The observation at time step t
            obs_tp1: The observation at time step t+1
            action: The action
            rew: The reward
            terminated: If the environment is terminated
            truncated: If the environment is truncated
            info: The information from the environment
        """
        points = [rew, obs_tp1[1], obs_tp1[2], info['static_freq_change'],  # type: ignore
                  info['penalty_energy'], info['temperature']]  
        for point, data_series in zip(points, self.data):
            data_series.append(point)
        self.t += 1

        xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t
        xrange = range(xmin, xmax)

        self.cur_plot[0][0].set_data(xrange, list(self.data[0]))  # reward

        self.cur_plot[1][0].set_data(xrange, list(self.data[1]))  # offset
        self.cur_plot[1][1].set_data(
            xrange, list(self.data[2]))  # target offset
        
        self.cur_plot[2][0].set_data(xrange, list(
            self.data[3]))  # static freq change
        self.cur_plot[2][1].set_data(
            xrange, list(self.data[2]))  # target offset
        
        self.cur_plot[3][0].set_data(
            xrange, list(self.data[4]))  # penalty energy
        
        self.cur_plot[4][0].set_data(xrange, list(self.data[5]))  # temperature

        for ax in self.ax:
            #ax.set_xlim(xmin, xmax)
            ax.relim()
            ax.autoscale_view()

        plt.pause(0.000001)


plotter = PlayPlot(30 * 5)

env = gym.make("gym_quarz/QuartzEnv-v5", render_mode='rgb_array')

env.reset()

frame = env.render()
video_size = frame.shape[:2]  # type: ignore
screen = pygame.display.set_mode(video_size)
fps = env.metadata.get("render_fps", 60)


mapping = {(pygame.K_ESCAPE,): 0, (pygame.K_DOWN,): 1,
           (pygame.K_SPACE,): 2, (pygame.K_END,): 3}
play(env, callback=plotter.callback,
     keys_to_action=mapping, noop=0)  # type: ignore
