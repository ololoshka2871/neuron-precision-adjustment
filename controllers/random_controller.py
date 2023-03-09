import numpy as np

from misc.common import limit

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

        def limitmp(x): return limit(x, -1.0, 1.0)
        def limitzp(x): return limit(x, 0.0, 1.0)

        return {
            'destination': list(map(limitmp, np.random.normal(0, 1.0, 2))),
            'power': limitzp(np.random.normal(0.5, 0.5)),
            'speed': limit(np.random.normal(0.5, 0.5), 0.05, 1.0),
            'self_evaluation': limitmp(np.random.normal(0, 1.0))
        }
