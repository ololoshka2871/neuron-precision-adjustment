#!/usr/bin/env python

from copy import deepcopy

import gymnasium as gym
import gym_quarz

from controllers.controller_v6_algo import AlgorithmicController as Controller

from constants_v6 import *


def sim_main() -> dict:
    env = gym.make("gym_quarz/QuartzEnv-v6", 
                   laser_power_relative=LASER_POWER_RELATIVE)

    obs, info = env.reset()

    ctrl = Controller(
        angle_change_step=info['horisontal_angle_step'], 
        angle_limit=info['max_angle'],
        freq_minimal_change_cooling=FREQ_MINIMAL_CHANGE_COOLING,
        fast_forward_steps=FAST_FORWARD_STEPS,
    )

    prev_obs, done = deepcopy(obs), False
    while not done:
        action = ctrl.sample_action(prev_obs, obs)
        prev_obs = obs
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    result = {
        'detected_rezonator_angle': ctrl.rezonator_angle,
        'actual_rezonator_angle': info['actual_rezonator_angle'],
        'adjusted_freq': obs[1],
        'adjust_target_freq': obs[2],
        'disbalance': info['disbalance'],
        'time_elapsed': info['time_elapsed'],
    }

    env.close()

    return result


if __name__ == '__main__':
    res = sim_main()

    print(f"""Done!
- Detected resonator angle: {res['detected_rezonator_angle']:.2f} (actual: {res['actual_rezonator_angle']:.2f}),
- Adjusted {res['adjusted_freq']:.3f} / {res['adjust_target_freq']:.3f} Hz ({res['adjusted_freq'] / res['adjust_target_freq'] * 100:.2f}%),
- Disbalance: {res['disbalance'] * 100:.2f}%.
- Total time: {res['time_elapsed']:.2f} s.
""")
