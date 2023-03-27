## https://stackoverflow.com/a/58628399
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Concatenate, Activation, Flatten

from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor


class PendulumProcessor(Processor):
    def process_reward(self, reward):
        return reward


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

    def __init__(self, nb_states, nb_actions, mem_limit=50000):
        #model = Sequential()
        #model.add(Flatten(input_shape=(1, states)))
        #model.add(Dense(24, activation='tanh'))
        #model.add(Dense(24, activation='linear'))
        #model.add(Dense(nb_actions, activation='tanh'))

        # Build all necessary models: V, mu, and L networks.
        V_model = Sequential()
        V_model.add(Flatten(input_shape=(nb_states,)))
        V_model.add(Dense(16))
        V_model.add(Activation('relu'))
        V_model.add(Dense(16))
        V_model.add(Activation('relu'))
        V_model.add(Dense(16))
        V_model.add(Activation('relu'))
        V_model.add(Dense(1))
        V_model.add(Activation('linear'))

        mu_model = Sequential()
        mu_model.add(Flatten(input_shape=(nb_states,)))
        mu_model.add(Dense(16))
        mu_model.add(Activation('relu'))
        mu_model.add(Dense(16))
        mu_model.add(Activation('relu'))
        mu_model.add(Dense(16))
        mu_model.add(Activation('relu'))
        mu_model.add(Dense(nb_actions))
        mu_model.add(Activation('linear'))

        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(nb_states,), name='observation_input')
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

        memory = SequentialMemory(limit=mem_limit, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
        processor = PendulumProcessor()
        super().__init__(
            nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
            memory=memory, nb_steps_warmup=100, random_process=random_process,
            gamma=.99, target_model_update=1e-3, processor=processor)
