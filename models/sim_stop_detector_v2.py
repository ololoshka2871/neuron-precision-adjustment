import time

import numpy as np

from matplotlib.pyplot import Axes

from models.rezonator_model import Metrics, RezonatorModel

from models.stop_condition import StopCondition


class SimStopDetector:
    """
    Класс принимает текущие метрики симуляции и принимает решение когда остановить симуляцию. Используется история метрик
    - Средняя скорость движения последние 2 меньше min_avg_speed
    - Средняя мощность 
    """

    def __init__(self,
                 timeout: float,
                 history_len_s: float,
                 min_path: float,
                 min_avg_speed: float,
                 min_laser_power: float,
                 max_temperature: float,
                 self_grade_epsilon=0.01,
                 start_timestamp=time.time()):
        """
        :param timeout: Безусловный таймаут [s]
        :param history_len_s: Длина истории метрик [s]
        :param min_path: Минимальное приведенное расстояние пройденое за время history_len_s - используется, чтобы определить что контроллер не двигает лазер
        :param min_avg_speed: Минимальная скорость движения - используется, чтобы определить что контроллер не двигает лазер [0..1]
        :param min_avg_power: Минимальная мощность лазера - используется, чтобы определить что сеть пытается работать с выключенным лазером [0..1]
        :param max_temperature: Максимальная температура - перегрев, стоп
        :param self_grade_epsilon: размер окресности точки 0 при попадании в которую значения нейрона самооценки останавливает симуляцию (0..1)
        """
        self._timeout = timeout
        self._history_len_s = history_len_s
        self._min_path = min_path
        self._min_avg_speed = min_avg_speed
        self._min_laser_power = min_laser_power
        self._max_temperature = max_temperature

        self._start_timestamp = start_timestamp

        self._timestamps = np.array([], dtype=float)
        self._path_history = np.array([], dtype=float)
        self._speed_history = np.array([], dtype=float)
        self._laser_power_history = np.array([], dtype=float)
        self._temperature_history = np.array([], dtype=float)
        self._self_grade_history = np.array([], dtype=float)

        self._self_grade_epsilon = self_grade_epsilon

        self._max_temperature = 0

    def _trimm_history_if_too_long(self, time: float) -> bool:
        if len(self._timestamps) > 0 and (time > self._timestamps[0] + self._history_len_s):
            self._timestamps = self._timestamps[1:]
            self._path_history = self._path_history[1:]
            self._speed_history = self._speed_history[1:]
            self._laser_power_history = self._laser_power_history[1:]
            self._temperature_history = self._temperature_history[1:]
            self._self_grade_history = self._self_grade_history[1:]
            return True
        return False

    def _add_metric(self, time: float, m: dict, T: float):
        self._timestamps = np.append(self._timestamps, time)

        self._path_history = np.append(self._path_history, m['Passed'])
        self._speed_history = np.append(self._speed_history, m['F'])
        self._laser_power_history = np.append(
            self._laser_power_history, m['S'])
        self._temperature_history = np.append(self._temperature_history, T)
        self._self_grade_history = np.append(
            self._self_grade_history, m['self_grade'])

        if T > self._max_temperature:
            self._max_temperature = T

    def __call__(self, t: float, rm: Metrics, mm: dict) -> StopCondition:
        """
        Акумулирует метрики и вынусоит суждение о том стоит ли остановить симуляцю
        Метрики симуляции:
        - Таймштамп
        - Текущая скорость движения (из интерполятора), приведенная
        - Текущая мощность лазера, приведенная
        - Значение нейрона самооценки
        - Температура резонатора [C] -> [K]
        """

        trimmed = self._trimm_history_if_too_long(t)
        self._add_metric(t, mm, rm['temperature'] + RezonatorModel.CELSUSS_TO_KELVIN)
        
        if len(self._timestamps) < 2:
            return StopCondition.NONE
        
        passed = self._timestamps[-1] - self._start_timestamp
        if passed > self._timeout:
            return StopCondition.TIMEOUT
        if trimmed and self._path_history.sum() < self._min_path:
            return StopCondition.STALL_MOVE
        if trimmed and self._speed_history.mean() < self._min_avg_speed:
            return StopCondition.STALL_SPEED
        if trimmed and self._laser_power_history.mean() < self._min_laser_power:
            return StopCondition.LOW_POWER
        if self._temperature_history.mean() / self._max_temperature > self._max_temperature:
            return StopCondition.OVERHEAT
        if abs(self._self_grade_history[-1]) < self._self_grade_epsilon:
            return StopCondition.SELF_STOP

        return StopCondition.NONE

    def summary(self) -> dict:
        return {
            'total_duration_rel': (self._timestamps[-1] - self._start_timestamp) / self._timeout,
            'self_grade': self._self_grade_history[-1],
            'max_temperature': self._max_temperature - RezonatorModel.CELSUSS_TO_KELVIN,
            'avg_speed': self._speed_history.mean(),
        }

    def plot_summary(self, ax: Axes):
        t = self._timestamps - self._start_timestamp
        ax.plot(t, self._path_history, 'co-', label='path_history')
        ax.plot(t, self._speed_history, 'bo-', label='speed_history')
        ax.plot(t, self._laser_power_history,
                'go-', label='laser_power_history')
        ax.plot(t, self._self_grade_history, 'yo-', label='self_grade_history')
        ax.plot(t, (self._temperature_history - self._temperature_history[0]) / self._max_temperature,
                'ro-', label='temperature_history')
        ax.legend()
