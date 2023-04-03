import numpy as np
import math

# https://stackoverflow.com/a/58628399
from keras.models import Sequential
from keras.layers import Dense, Flatten

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


class LaserProcessor(Processor):
    def __init__(self, history_len=32) -> None:
        super().__init__()

        self._history_len = history_len
        self._history = np.array([])

    @property
    def history_size(self) -> int:
        return self._history_len * 2

    def transform_observation_space(self, observation_space):
        return (observation_space[0] - 2 + self._history_len * 2,)

    def process_observation(self, observation):
        y, freq_change, freq_change_target, sym_time = observation

        if freq_change == 0:
            freq_change = 1e-1
        freq_change_rel_inv = math.log(freq_change, 15.0)
        freq_chang_target_inv = math.log(freq_change_target, 15.0)  # Чтобы было меньше 1

        if self._history.size == 0:
            self._history = np.array(
                [y, freq_change_rel_inv] * self._history_len)
        else:
            self._history = np.append(
                self._history, [y, freq_change_rel_inv])
            self._history = self._history[2:]

        return np.array([freq_chang_target_inv, sym_time, *self._history])

    def process_step(self, observation, reward, done, info):
        if done:
            self._history = np.array([])  # reset history

        return self.process_observation(observation), reward, done, info

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


class DQNController(DQNAgent):
    """
    Контроллер - нейронная сеть
    """

    def __init__(self,
                 obs_space, action_space,
                 history_len=32,
                 nb_steps_warmup=5000,
                 gamma=0.99, target_model_update=10000,
                 mem_limit=5000000,
                 train_interval=4, delta_clip=1.0
                 ):
        WINDOW_LENGTH = 4

        nb_actions = action_space.n
        processor = LaserProcessor(history_len=history_len)

        input_shape = (WINDOW_LENGTH,) + \
            processor.transform_observation_space(obs_space.shape)
        actor_neurons = 16 + processor.history_size

        # observation -> action
        actor = Sequential()
        actor.add(Flatten(input_shape=input_shape, name='Actor'))
        actor.add(Dense(actor_neurons, activation='linear'))
        actor.add(Dense(actor_neurons, activation='relu'))
        actor.add(Dense(actor_neurons, activation='relu'))
        actor.add(Dense(actor_neurons, activation='relu'))
        actor.add(Dense(actor_neurons, activation='relu'))
        actor.add(Dense(nb_actions, activation='linear'))
        print(actor.summary())

        memory = SequentialMemory(limit=mem_limit, window_length=WINDOW_LENGTH)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                      attr='eps', value_max=1., 
                                      value_min=.1, value_test=.05,
                                      nb_steps=1000000)

        super().__init__(
            model=actor,
            nb_actions=nb_actions, policy=policy, memory=memory,
            processor=processor, nb_steps_warmup=nb_steps_warmup,
            gamma=gamma, target_model_update=target_model_update,
            train_interval=train_interval, delta_clip=delta_clip,
        )
