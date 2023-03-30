import numpy as np

# https://stackoverflow.com/a/58628399
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Concatenate, Activation, Flatten

from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor


class LaserProcessor(Processor):
    def __init__(self, history_len=32) -> None:
        super().__init__()

        self._history_len = history_len
        self._measure_history = np.array([])

    @property
    def history_len(self) -> int:
        return self._history_len

    def transform_observation_space(self, observation_space):
        return (observation_space[0] + self._history_len - 1,)

    def process_action(self, action):
        """
        action[0] - признак действия [0..1]
        action[1] - x [-1..1]
        action[2] - y [-1..1]
        action[3] - speed [0..1]
        Там добавляется шум, нужно клипать
        """
        act = np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0)
        x = np.clip(action[1], -1.0, 1.0)
        y = np.clip(action[2], -1.0, 1.0)
        f = np.clip((action[3] + 1.0) / 2.0, 0.0, 1.0)
        return np.array([act, x, y, f])

    def process_observation(self, observation):
        pos = observation[:2]
        F = observation[2]
        freq_change = observation[3]
        freq_change_target = observation[4]
        sym_time = observation[5]

        freq_change_rel = freq_change / freq_change_target
        freq_chang_target_inv = 1.0 / freq_change_target  # Чтобы было меньше 1

        if self._measure_history.size == 0:
            self._measure_history = np.repeat(
                freq_change_rel, self._history_len)
        else:
            self._measure_history = np.append(
                self._measure_history, freq_change_rel)
            self._measure_history = self._measure_history[1:]

        return np.array(
            [*pos, F, freq_chang_target_inv, sym_time, *self._measure_history])

    def process_step(self, observation, reward, done, info):
        if done:
            self._measure_history = np.array([])  # reset history

        return self.process_observation(observation), reward, done, info


class NNController(NAFAgent):
    """
    Контроллер - нейронная сеть
    Входы:
        - position: spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        - power: spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        - current_frequency_offset: spaces.Box(-1e+6, 1e+6, shape=(1,), dtype=np.float32)
        - adjust_target: spaces.Box(-1000, 1000, shape=(1,), dtype=np.float32)
        - time: spaces.Box(0.0, 100.0, shape=(1,), dtype=np.float32)

    Выходы:
        - "move": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32): Передвинуть лазер на заданное расстояние
        - "set_power": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32): Изменить мощность лазера
        - wait: spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32): Ожидаение
        - end: spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32): Закончить эпизод
    """

    def __init__(self, obs_space, action_space,
                 nb_steps_warmup=100,
                 gamma=0.99, target_model_update=1e-3,
                 theta=0.15, mu=0.0, sigma=0.3,
                 batch_size=32,
                 history_len=32,
                 mem_limit=5000000):
        nb_actions = action_space.shape[0]
        processor = LaserProcessor(history_len=history_len)

        input_shape = (1,) + \
            processor.transform_observation_space(obs_space.shape)
        dens_neurons = 16 + processor.history_len

        # Build all necessary models: V, mu, and L networks.
        # observation -> V
        V_model = Sequential()
        V_model.add(Flatten(input_shape=input_shape, name='V'))
        V_model.add(Dense(dens_neurons, activation='tanh'))
        V_model.add(Dense(dens_neurons, activation='linear'))
        V_model.add(Dense(dens_neurons, activation='linear'))
        V_model.add(Dense(1, activation='linear'))
        print(V_model.summary())

        # observation -> action
        mu_model = Sequential()
        mu_model.add(Flatten(input_shape=input_shape, name='mu'))
        mu_model.add(Dense(dens_neurons, activation='tanh'))
        mu_model.add(Dense(dens_neurons, activation='linear'))
        mu_model.add(Dense(dens_neurons, activation='linear'))
        mu_model.add(Dense(nb_actions, activation='tanh'))
        print(mu_model.summary())

        # observation, action -> L ((nb_actions^2 + nb_actions) // 2 outputs)
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=input_shape, name='observation_input')
        x = Concatenate()([action_input, Flatten()(observation_input)])
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
        x = Activation('linear')(x)
        L_model = Model(inputs=[action_input, observation_input], outputs=x)
        print(L_model.summary())

        memory = SequentialMemory(limit=mem_limit, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(
            theta=theta, mu=mu, sigma=sigma, size=nb_actions)

        super().__init__(
            nb_actions=nb_actions, batch_size=batch_size,
            V_model=V_model, L_model=L_model, mu_model=mu_model,
            memory=memory, nb_steps_warmup=nb_steps_warmup, random_process=random_process,
            gamma=gamma, target_model_update=target_model_update, processor=processor)
