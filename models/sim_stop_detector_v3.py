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
                 max_temperature: float,
                 self_grade_epsilon=0.01,
                 start_timestamp=0.0):
        """
        :param timeout: Безусловный таймаут [s]
        :param history_len_s: Длина истории метрик [s]
        :param max_temperature: Максимальная температура - перегрев, стоп
        :param self_grade_epsilon: размер окресности точки 0 при попадании в которую значения нейрона самооценки останавливает симуляцию (0..1)
        :param start_timestamp: Время начала симуляции
        """
        self._timeout = timeout
        self._history_len_s = history_len_s
        self._temperature_limit = max_temperature

        self._start_timestamp = start_timestamp

        self._timestamps = np.array([], dtype=float)
        self._path_history = np.array([], dtype=float)
        self._speed_history = np.array([], dtype=float)
        self._laser_power_history = np.array([], dtype=float)
        self._temperature_history = np.array([], dtype=float)
        self._self_grade_history = np.array([], dtype=float)
        self._freq_history = np.array([], dtype=float)

        self._self_grade_epsilon = self_grade_epsilon

        self._max_temperature = 0.0
        self._path_accum = 0.0

    def _trimm_history_if_too_long(self, time: float) -> bool:
        if len(self._timestamps) > 0 and (time > self._timestamps[0] + self._history_len_s):
            self._timestamps = self._timestamps[1:]
            self._path_history = self._path_history[1:]
            self._speed_history = self._speed_history[1:]
            self._laser_power_history = self._laser_power_history[1:]
            self._temperature_history = self._temperature_history[1:]
            self._self_grade_history = self._self_grade_history[1:]
            self._freq_history = self._freq_history[1:]
            return True
        return False

    def _add_metric(self, time: float, m: dict, T: float, F: float):
        self._timestamps = np.append(self._timestamps, time)

        self._path_history = np.append(self._path_history, m['Passed'])
        self._speed_history = np.append(self._speed_history, m['F'])
        self._laser_power_history = np.append(
            self._laser_power_history, m['S'])
        self._temperature_history = np.append(self._temperature_history, T)
        self._self_grade_history = np.append(
            self._self_grade_history, m['self_grade'])
        self._freq_history = np.append(self._freq_history, F)

        if T > self._max_temperature:
            self._max_temperature = T

    def get_time_limit(self) -> float:
        return self._timeout

    def __call__(self, t: float, rm: Metrics, mm: dict, F_measured: float) -> StopCondition:
        """
        Акумулирует метрики и вынусоит суждение о том стоит ли остановить симуляцю
        Метрики симуляции:
        - Таймштамп
        - Текущая скорость движения (из интерполятора), приведенная
        - Текущая мощность лазера, приведенная
        - Значение нейрона самооценки
        - Температура резонатора [C] -> [K]
        """

        self._path_accum += mm['Passed']

        trimmed = self._trimm_history_if_too_long(t)
        self._add_metric(t, mm, rm['temperature'] +
                         RezonatorModel.CELSUSS_TO_KELVIN, F_measured)

        if len(self._timestamps) < 2:
            return StopCondition.NONE

        passed = self._timestamps[-1] - self._start_timestamp
        if passed > self._timeout:
            return StopCondition.TIMEOUT
        if self._temperature_history.mean() > self._temperature_limit:
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
            'total_path_len': self._path_accum,
            'avg_laser_power': self._laser_power_history.mean(),
        }

    def plot_summary(self, ax: Axes):
        t = self._timestamps - self._start_timestamp
        ax.plot(t, SimStopDetector._normalize(
            self._path_history), 'co-', label='path_history')
        ax.plot(t, self._speed_history, 'bo-', label='speed_history')
        ax.plot(t, self._laser_power_history,
                'go-', label='laser_power_history')
        ax.plot(t, self._self_grade_history, 'yo-', label='self_grade_history')
        ax.plot(t, SimStopDetector._normalize(
            self._temperature_history), 'ro-', label='temperature_history')
        ax.plot(t, SimStopDetector._normalize(
            self._freq_history), 'ko-', label='freq_history')
        ax.legend()
        ax.grid()

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        size = x.max() - x.min()
        if size == 0.0:
            return x
        else:
            return (x - x.min()) / size
