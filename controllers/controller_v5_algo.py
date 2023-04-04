import numpy as np
import math

#class LaserProcessor(Processor):
#    NORM_FACTOR = 50.0
#
#    def __init__(self, history_len=32) -> None:
#        super().__init__()
#
#        self._history_len = history_len
#        self._history = np.array([])
#
#    @property
#    def history_size(self) -> int:
#        return self._history_len * 2
#
#    def transform_observation_space(self, observation_space):
#        return (observation_space[0] - 2 + self._history_len * 2,)
#
#    def process_observation(self, observation):
#        y, freq_change_clipped, freq_change_target, sym_time = observation
#
#        freq_change_clipped = np.clip(freq_change_clipped, -LaserProcessor.NORM_FACTOR,
#                                      LaserProcessor.NORM_FACTOR)
#        freq_change_normed = freq_change_clipped / LaserProcessor.NORM_FACTOR
#        freq_chang_target_normed = freq_change_target / LaserProcessor.NORM_FACTOR
#
#        if self._history.size == 0:
#            self._history = np.array(
#                [y, freq_change_normed] * self._history_len)
#        else:
#            self._history = np.append(
#                self._history, [y, freq_change_normed])
#            self._history = self._history[2:]
#
#        return np.array([freq_chang_target_normed, sym_time, *self._history])
#
#    def process_step(self, observation, reward, done, info):
#        if done:
#            self._history = np.array([])  # reset history
#
#        return self.process_observation(observation), reward, done, info
#
#    def process_reward(self, reward):
#        return np.clip(reward, -1., 1.)


class AlgorithmicController:
    """
    Контроллер алгоритмический
    """

    def __init__(self, obs_space, action_space):
        self._obs_space = obs_space
        self._action_space = action_space

    def sample_action(self, prev_observation, observation):
        v = self._action_space.sample()
        if v == 3:
            v = 0
        return v
