import numpy as np


class RandomController:
    """
    Заглушка для контроллера, вмето вызодных параметров возвращает случайные значения в заданных диопазонах
    - Цель движения: tuple[x, y]: -1.0..1.0
    - Скорость перемещения: 0.0..1.0
    - Выходная мощность лазера: 0.0..1.0
    - Самооценка: -1.0..1.0
    """

    def __init__(self) -> None:
        pass

    def update(self, input: dict):
        """
        Функция обновляет состояние контроллера
        """

        limitmp = lambda x: RandomController._limit(x, -1.0, 1.0)
        limitzp = lambda x: RandomController._limit(x, 0.0, 1.0)

        return {
            'destination': list(map(limitmp, np.random.normal(0, 1.0, 2))),
            'power': limitzp(np.random.normal(0.5, 0.5)),
            'speed': RandomController._limit(np.random.normal(0.5, 0.5), 0.05, 1.0),
            'self_evaluation': limitmp(np.random.normal(0, 1.0))
        }
    
    @staticmethod
    def _limit(v: float, _min: float, _max: float) -> float:
        return max(_min, min(v, _max))


class NeuralNetworkController:
    pass