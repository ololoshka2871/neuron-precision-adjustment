from keras.layers import Flatten, Dense
from keras.models import Sequential

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


class NNController(DQNAgent):
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

    def __init__(self, states, actions, mem_limit=50000, warmup=10, target_model_update=1e-2):
        model = Sequential()

        model.add(Flatten(input_shape=(1, states)))
        model.add(Dense(24, activation='tanh'))
        model.add(Dense(24, activation='linear'))
        model.add(Dense(actions, activation='tanh'))

        policy = BoltzmannQPolicy()
        memory = SequentialMemory(limit=mem_limit, window_length=1)

        super().__init__(model=model,
                         memory=memory,
                         policy=policy,
                         nb_actions=actions,
                         nb_steps_warmup=warmup,
                         target_model_update=target_model_update)
