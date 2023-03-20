
import numpy as np

from misc.common import gen_sigmoid, normal_dist

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
                 grade_weights: np.ndarray = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])):
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
            - 7 Бонус за дину пройденнго пути - больше - лучше
            - 8 Бонус за остаток энергии - больше - лучше (лучше если энергия есть в запасе, но не более 30%)
            - 9 Оценка за причину остановки - больше - лучше
        """

        # относительная дистанция от текущей частоты до желаемой
        freq_change = rezonator_metrics['static_freq_change']
        freq_target_distance_rel = (
            self._dest_freq_ch - freq_change) / self._dest_freq_ch

        db = abs(rezonator_metrics['disbalance'])

        sigmoid_grade = gen_sigmoid(A=2.0, k=5.0, x_offset_to_right=0, vertical_shift=-1.0)

        path = sim_metrics['total_path_len']

        adjust_grade = sigmoid_grade(freq_target_distance_rel) if freq_target_distance_rel > 0.0 else -freq_target_distance_rel * 2.0
        penalty = self._f_penalty(rezonator_metrics['penalty_energy'])
        disbalance = sigmoid_grade(db) if db > 0 else 1.0
        self_grade_accuracy = 1.0 - (sim_metrics['self_grade'] - freq_target_distance_rel)
        temp_rel = sim_metrics['max_temperature'] / self._max_temperature
        speed_avg = sim_metrics['avg_speed']
        time_rel = sim_metrics['total_duration_rel']
        path_bonus = path * ((1.0 - adjust_grade * 0.25))
        energy_left = normal_dist(sim_metrics['energy_relative'], mean=0.3, sd=0.10)
        stop_condition_grade = self._grade_stop_condition[stop_condition]

        #w = np.array([
        #    adjust_grade,  # настройка
        #    self._f_penalty(rezonator_metrics['penalty_energy']),  # штраф за попадание куда не надо
        #    sigmoid_grade(db) if db > 0 else 1.0,  # Если дисбаланса вообще нет - скорее всего нет и настройки -> доп штраф!
        #    1.0 - (sim_metrics['self_grade'] - freq_target_distance_rel),  # точность самооценки
        #    sim_metrics['max_temperature'] / self._max_temperature,  # температура
        #    sim_metrics['avg_speed'],  # скорость
        #    sim_metrics['total_duration_rel'],  # время
        #    path * ((1.0 - adjust_grade * 0.25)),  # длина пути
        #    normal_dist(sim_metrics['energy_relative'], mean=0.3, sd=0.10),  # остаток энергии
        #    self._grade_stop_condition[stop_condition]  # причина остановки
        #])

        w = np.array([0.0] * 10)

        # Вообще не работает
        ## Самый важный параметр - настройка
        #w[0] = adjust_grade
        #if adjust_grade < 0.3:
        #    # Если настройка хорошая - то штраф за попадание куда не надо уже учитывается
        #    w[1] = penalty
        #    if penalty < 0.1:
        #        # Если штрафа нет - то дисбаланс уже учитывается
        #        w[2] = disbalance
        #        if disbalance < 0.3:
        #            # Если дисбаланс мал - то уже учитывается самооценка, температура, скорость, время
        #            w[3] = self_grade_accuracy
        #            w[4] = temp_rel
        #            w[5] = speed_avg
        #            w[6] = time_rel
        #            if time_rel < 0.8 and time_rel > 0.3:
        #                # Если время симуляции в пределах 30% - то уже учитывается длина пути и остаток энергии
        #                w[7] = path_bonus
        #                w[8] = energy_left
        #                if energy_left > 0.0:
        #                    # Если остаток энергии есть - то уже учитывается причина остановки
        #                    w[9] = stop_condition_grade

        # На основе пройденного пути и времени симуляции
        w[7] = path_bonus
        if path > 5.0 and time_rel < 0.8 and time_rel > 0.3:
            w[6] = time_rel
            w[8] = energy_left
            w[1] = penalty
            if penalty < 0.1:
                w[0] = adjust_grade
                if adjust_grade < 0.3:
                    w[2] = disbalance
                    if disbalance < 0.3:
                        w[3] = self_grade_accuracy
                        w[4] = temp_rel
                        w[5] = speed_avg
                        w[9] = stop_condition_grade
        
        return (w * self._grade_weights).sum(), w
