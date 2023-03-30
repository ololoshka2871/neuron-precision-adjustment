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
        self._history = np.array([])

    @property
    def history_size(self) -> int:
        return self._history_len * 4

    def transform_observation_space(self, observation_space):
        return (observation_space[0] - 4 + self._history_len * 4,)

    def process_action(self, action):
        """
        - 0: Вероятность сделать движение
        - 1: x - Координата x, если move, иначе power или время ожидания
        - 2: y - Координата y
        - 3: F - Скорость перемещения
        - 4: Вероятность установить мощность лазера
        - 5: S - Мощность лазера
        - 6: Вероятность ожидания
        - 7: T - Время ожидания
        - 8: Вероятность закончить эпизод
        Там добавляется шум, нужно клипать операнды
        """
        move = action[0]
        x = action[1]
        y = action[2]
        f = np.clip((action[3] + 1.0) / 2.0, 0.0, 1.0)
        set_power = action[4]
        S = np.clip((action[5] + 1.0) / 2.0, 1.0e-3, 1.0)
        wait = action[6]
        T = np.clip((action[5] + 1.0) / 2.0, 0.0, 1.0)
        end = action[7]
        return np.array([move, x, y, f, set_power, S, wait, T, end])

    def process_observation(self, observation):
        pos = observation[:2]
        F = observation[2]
        freq_change = observation[3]
        freq_change_target = observation[4]
        sym_time = observation[5]

        freq_change_rel = freq_change / freq_change_target
        freq_chang_target_inv = 1.0 / freq_change_target  # Чтобы было меньше 1

        if self._history.size == 0:
            self._history = np.array(
                [*pos, F, freq_change_rel] * self._history_len)
        else:
            self._history = np.append(
                self._history, [*pos, F, freq_change_rel])
            self._history = self._history[4:]

        return np.array([freq_chang_target_inv, sym_time, *self._history])

    def process_step(self, observation, reward, done, info):
        if done:
            self._history = np.array([])  # reset history

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
        v_mu_dens_neurons = 16 + processor.history_size
        L_dense_neurons = 32 + processor.history_size // 2

        # Build all necessary models: V, mu, and L networks.
        # observation -> V
        V_model = Sequential()
        V_model.add(Flatten(input_shape=input_shape, name='V'))
        V_model.add(Dense(v_mu_dens_neurons, activation='tanh'))
        V_model.add(Dense(v_mu_dens_neurons, activation='linear'))
        V_model.add(Dense(v_mu_dens_neurons, activation='linear'))
        V_model.add(Dense(1, activation='linear'))
        print(V_model.summary())

        # observation -> action
        mu_model = Sequential()
        mu_model.add(Flatten(input_shape=input_shape, name='mu'))
        mu_model.add(Dense(v_mu_dens_neurons, activation='tanh'))
        mu_model.add(Dense(v_mu_dens_neurons, activation='linear'))
        mu_model.add(Dense(v_mu_dens_neurons, activation='linear'))
        mu_model.add(Dense(nb_actions, activation='tanh'))
        print(mu_model.summary())

        # observation, action -> L ((nb_actions^2 + nb_actions) // 2 outputs)
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=input_shape, name='observation_input')
        x = Concatenate()([action_input, Flatten()(observation_input)])
        x = Dense(L_dense_neurons)(x)
        x = Activation('relu')(x)
        x = Dense(L_dense_neurons)(x)
        x = Activation('relu')(x)
        x = Dense(L_dense_neurons)(x)
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
