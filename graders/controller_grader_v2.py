
import numpy as np

from models.rezonator_model import Metrics
from models.stop_condition import StopCondition


class ControllerGrager:
    """
    Преобразует результат симуляции в оценку работы
    """

    def __init__(self,
                 dest_freq_ch: float,
                 grade_stop_condition: dict[StopCondition, float] = {
                     StopCondition.TIMEOUT: -0.5,
                     StopCondition.STALL_SPEED: -0.3,
                     StopCondition.STALL_MOVE: -0.7,
                     StopCondition.LOW_POWER: -0.6,
                     StopCondition.OVERHEAT: -0.2,
                     StopCondition.SELF_STOP: 0.2,
                     StopCondition.NO_ENERGY: -1.0,
                     StopCondition.NONE: 0.0
                 },
                 f_penalty=lambda x: x,
                 max_temperature=1000.0,
                 grade_weights: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])):
        """
        :param dest_freq_ch: Зелаемое изменение частоты [Hz]
        :param f_penalty: Функция, преобразующая накопленную штрафную энергию в значение [0..1]
        :param max_temperature: максимальная температура резонатора [K]
        :param grade_stop_condition: функция оценки причины остановки симуляции
        """
        self._dest_freq_ch = dest_freq_ch
        self._f_penalty = f_penalty
        self._max_temperature = max_temperature
        self._grade_stop_condition = grade_stop_condition
        self._grade_weights = grade_weights

    def get_grade(self, rezonator_metrics: Metrics, sim_metrics: dict, stop_condition: StopCondition):
        """
        Оценка:
            - 0 Относительная дистанция до целевой частоты - меньше - лучше
            - 1 Относительный штраф за попадание куда не надо - меньше - лучше
            - 2 Относителдьный диссбаланс - меньше - лучше
            - 3 Точность самооценки - больше - лучше
            - 4 Максимальная достигнутая температура - меньше - лучше
            - 5 Средняя скорость движения - больше - лучше
            - 6 Относительное время симуляции - меньше - лучше
            - 7 Оценка за причину остановки - больше - лучше
        """

        # относительная дистанция от текущей частоты до желаемой
        freq_change = rezonator_metrics['static_freq_change']
        freq_target_distance_rel = (
            self._dest_freq_ch - freq_change) / self._dest_freq_ch

        db = abs(rezonator_metrics['disbalance'])

        w = np.array([
            freq_target_distance_rel if freq_target_distance_rel > 0.0 else -freq_target_distance_rel * 2.0, # настройка
            self._f_penalty(rezonator_metrics['penalty_energy']), # штраф за попадание куда не надо
            db if db > 0 else 1.0,  # Если дисбаланса вообще нет - скорее всего нет и настройки -> доп штраф!
            1.0 - (sim_metrics['self_grade'] - freq_target_distance_rel), # точность самооценки
            sim_metrics['max_temperature'] / self._max_temperature,  # температура
            sim_metrics['avg_speed'],  # скорость
            sim_metrics['total_duration_rel'],  # время
            self._grade_stop_condition[stop_condition]  # причина остановки
        ])
        
        return (w * self._grade_weights).sum(), w
