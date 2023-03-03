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

    def update(self):
        """
        Функция обновляет состояние контроллера
        """
        return {
            'destination': np.random.normal(-1.0, 1.0, 2),
            'power': np.random.normal(0.0, 1.0),
            'speed': np.random.normal(0.0, 1.0),
            'self_evaluation': np.random.normal(-1.0, 1.0)
        }
