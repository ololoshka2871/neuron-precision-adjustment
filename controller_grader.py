
from rezonator_model import Metrics
from sim_stop_detector import StopCondition


class ControllerGrager:
    """
    Преобразует результат симуляции в оценку работы
    """

    def __init__(self,
                 dest_freq_ch: float,
                 f_penalty=lambda x: x,
                 max_temperature=1000.0,
                 grade_stop_condition=lambda sc: 0.0):
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

    def get_grade(self, rezonator_metrics: Metrics, sim_metrics: dict, stop_condition: StopCondition) -> tuple:
        """
        Оценка:
            - Относительная дистанция до целевой частоты - меньше - лучше
            - Относителдьный диссбаланс - меньше - лучше
            - Относительный штраф за попадание куда не надо - меньше - лучше
            - Относительное время симуляции - меньше - лучше
            - Точность самоошенки - больше - лучше
            - Максимальная достигнутая температура - меньше - лучше
            - Средняя скорость движения - больше - лучше
            - Оценка за причину остановки - больше - лучше
        """

        # относительная дистанция от текущей частоты до желаемой
        freq_target_distance_rel = (
            self._dest_freq_ch - rezonator_metrics['static_freq_change']) / (2.0 * self._dest_freq_ch)

        return (freq_target_distance_rel,
                rezonator_metrics['disbalance'],
                self._f_penalty(rezonator_metrics['penalty_energy']),
                sim_metrics['total_duration_rel'],
                1.0 - (sim_metrics['self_grade'] - freq_target_distance_rel),
                sim_metrics['max_temperature'] / self._max_temperature,
                sim_metrics['avg_speed'],
                self._grade_stop_condition(stop_condition))
