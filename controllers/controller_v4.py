import numpy as np

## https://stackoverflow.com/a/58628399
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Concatenate, Activation, Flatten, Lambda

from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor


class LaserProcessor(Processor):
    def process_reward(self, reward):
        return reward
    
    #def process_observation(self, observation):
    #    return observation[0]

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


class NNController(NAFAgent):
    """
    Контроллер - нейронная сеть
    Входы:
        - position: spaces.Box(-1.0, 1.0, shape=(2,), dtype=float)
        - power: spaces.Box(0.0, 1.0, shape=(1,), dtype=float)
        - current_frequency_offset: spaces.Box(-1e+6, 1e+6, shape=(1,), dtype=float)
        - adjust_target: spaces.Box(-1000, 1000, shape=(1,), dtype=float)

    Выходы:
        - "move": spaces.Box(-1.0, 1.0, shape=(4,), dtype=float): Передвинуть лазер на заданное расстояние
        - "set_power": spaces.Box(0.0, 1.0, shape=(2,), dtype=float): Изменить мощность лазера
        - wait: spaces.Box(0.0, 1.0, shape=(2,), dtype=float): Ожидаение
        - end: spaces.Box(0.0, 1.0, shape=(1,), dtype=float): Закончить эпизод
    """

    def __init__(self, obs_space, action_space, mem_limit=50000):
        nb_actions = action_space.shape[0]

        # Build all necessary models: V, mu, and L networks.
        # observation -> V
        V_model = Sequential()
        V_model.add(Flatten(input_shape=(1,) + obs_space.shape))
        V_model.add(Dense(16, activation='relu'))
        V_model.add(Dense(16, activation='relu'))
        V_model.add(Dense(16, activation='relu'))
        V_model.add(Dense(1, activation='linear'))
        print(V_model.summary())

        # observation -> action
        mu_model = Sequential()
        mu_model.add(Flatten(input_shape=(1,) + obs_space.shape))
        mu_model.add(Dense(16, activation='relu'))
        mu_model.add(Dense(16, activation='relu'))
        mu_model.add(Dense(16, activation='relu'))
        mu_model.add(Dense(nb_actions, activation='tanh'))
        print(mu_model.summary())

        # observation, action -> L ((nb_actions^2 + nb_actions) // 2 outputs)
        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + obs_space.shape, name='observation_input')
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
        random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
        processor = LaserProcessor()
        super().__init__(
            nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
            memory=memory, nb_steps_warmup=100, random_process=random_process,
            gamma=.99, target_model_update=1e-3, processor=processor)
