
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
                     StopCondition.NONE: 0.0
                 },
                 f_penalty=lambda x: x,
                 max_temperature=1000.0):
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
            - Относителдьный диссбаланс - меньше (по модулю) - лучше
            - Относительный штраф за попадание куда не надо - меньше - лучше
            - Относительное время симуляции - меньше - лучше
            - Точность самоошенки - больше - лучше
            - Максимальная достигнутая температура - меньше - лучше
            - Средняя скорость движения - больше - лучше
            - Оценка за причину остановки - больше - лучше
        """

        # относительная дистанция от текущей частоты до желаемой
        freq_target_distance_rel = (
            self._dest_freq_ch - rezonator_metrics['static_freq_change']) / self._dest_freq_ch
        
        db = abs(rezonator_metrics['disbalance'])

        return (freq_target_distance_rel,
                db if db > 0 else 1.0, # Если дисбаланса вообще нет - скорее всего нет и настройки -> доп штраф!
                self._f_penalty(rezonator_metrics['penalty_energy']),
                sim_metrics['total_duration_rel'],
                1.0 - (sim_metrics['self_grade'] - freq_target_distance_rel),
                sim_metrics['max_temperature'] / self._max_temperature,
                sim_metrics['avg_speed'],
                self._grade_stop_condition[stop_condition])
