
import numpy as np

import matplotlib.pyplot as plt

from misc.common import limit


class ManualController:
    """
    Ручной контроллер
    Входы:
        - freq_history - История измерений freq_history_size
        - move_history - История перемещений вида (dest_pos, S, F) длиной move_history_size
        - time - Относительное время до таймаута (0.0..1.0)

    Выходы:
        - Желаемая цель движения: tuple[x, y]: -1.0..1.0 (клик)
        - Желаемая скорость перемещения: 0.0..1.0 (рандом)
        - Желаемая выходная мощность лазера: 0.0..1.0 (рандом)
        - Самооценка: 0.0..1.0 (рандом)
    """

    @staticmethod
    def map_zero_one(v: float) -> float:
        return (1.0 + v) / 2.0
              
    def update(self, input: dict) -> dict:
        """
        Функция принимает на вход текущее состояние и симуляции, рисует его в GUI 
            и возвращает команду по клику мыши
        """

        while True:
            click = plt.ginput(1, timeout=1, show_clicks=False)
            if not click:
                continue

            click = click[0]
            return {
                'destination': click,
                'power': limit(np.random.normal(0.5, 0.2), 0.0, 1.0),
                'speed': limit(np.random.normal(0.5, 0.2), 1e-10, 1.0),
                'self_grade': np.random.uniform(0.0, 1.0)
            }
