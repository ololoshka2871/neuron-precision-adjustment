import numpy as np

from misc.queue import Queue
from models.rezonator_model import RezonatorModel, Zone, ModelView
from models.movement import Movment
from misc.work_zone import WorkZone, Rect


class Simulator:
    """
    Симуляция обработки лазером резонатора v2
    Отличие от v1:
    - Вместо расчета положения станка по точкам можно сразу расчитать путь с указанным шагом и скоростью
    - Если текущее заданное перемещение оказывается дольше, чем времение до следующего измерения, 
        то расчитывается расчитывается кусок до изверения, вычислятся новое измерение и продолжается движение до конца 
        или следующего измерения. Таким образом избагается множественный вызов расчета нейронной сети по 1 шагу
    - Нейронной сети на вход передаются не просто координаты текущего положения станка, а история последних move_history_len
        положений в виде наборов {dest_ps, S, F}.
    - Благодоря тому факту, что ели движение запущено его нельзя прервать или изменить цель то просто соберем историю измнений
        частоты и не будем считать нейронку каждый шаг симуляции, а будем считать только по окончанию движения, чтобы получить 
        новую команду на движение
    """

    def __init__(self,
                 rezonator_model: RezonatorModel,
                 controller_v2,
                 global_center: tuple[float, float],
                 initial_freq_diff=0.5,
                 max_f=1000.0,
                 freqmeter_period=0.4,
                 modeling_period=0.01,
                 laser_power_max_s=255.0,
                 freq_history_size=10,
                 move_history_len=10):
        """
        :param rezonator: Резонатор
        :param controller_v2: Контроллер (API v2)
        :param global_center: Координаты центра рабочей зоны в глобальной системе координат
        :param initial_freq_diff: Для от максимально-возможной перестройки частоты для этой симуляции
        :param max_f: Максимальная скорость перемещения станка
        :param freqmeter_period: Период измерения частоты, [с]
        :param modeling_period: Период моделирования движения и растройки резонатора, [с]
        :param laser_power_max_s: Максимальная мощность лазера [W]
        :param freq_history_size: Размер истории измерений частоты
        """
        self._rezonator_model = rezonator_model
        self._controller = controller_v2
        self._freqmeter_period = freqmeter_period
        self._modeling_period = modeling_period

        self._work_zone = WorkZone(Rect.from_rezonator(
            RezonatorModel.REZONATOR, global_center),
            laser_power_max_s, max_f)

        self._period_accum = 0.0
        self._next_mesure_after = 0.0

        self._movement = Movment()
        self._curremt_pos_global = self._work_zone.map_to_global((0, 1.0))

        self._initial_freq_diff = initial_freq_diff

        self._measure_diff_history = Queue(shape=(freq_history_size, 1))
        self._measure_diff_history.enqueue(
            np.repeat(initial_freq_diff, freq_history_size))

        self._move_history = Queue(shape=(move_history_len, 4))

        # (dest_pos_x, dest_pos_y, S, F)
        self._move_history.enqueue(
            np.reshape(
                np.repeat(
                    np.array([0.0, 1.0, 0.0, 0.0]), move_history_len),
                newshape=(4, move_history_len)).T)

    def perform_modeling(self, stop_detector, input_display=lambda input: None):
        """
        Выполнение симуляции
        :param stop_detector: Детектор условия остановки симуляции
        :return: Причина остановки симуляции
        """

        while True:
            controller_input = dict(
                time=self._period_accum,
                freq_history=self._measure_diff_history.peek_all(),
                move_history=self._move_history.peek_all(),
            )
            input_display(controller_input)
            command = self._controller.update(controller_input)

            traectory = self._movement.interpolate_move(
                src=self._curremt_pos_global,
                dst=self._work_zone.map_to_global(command['destination']),
                speed=self._work_zone.map_f_to_global(command['speed']),
                time_step=self._modeling_period
            )

            cmd_s = command['power']

            ts = 0.0  # заглушка
            for pos_x, pos_y, ts in zip(*traectory):
                self._next_mesure_after -= self._modeling_period
                if self._next_mesure_after <= 0.0:
                    self._next_mesure_after = self._freqmeter_period

                    # Сдвигаем историю измерений частоты и добавляем новое измерение
                    self._measure_diff_history.dequeue()
                    m = self._rezonator_model.get_metrics()
                    self._measure_diff_history.enqueue(
                        np.array([self._initial_freq_diff - m['freq_change']]))

                model_pos = self._work_zone.map_relative_to_local(
                    self._work_zone.map_from_global((pos_x, pos_y)))

                match ModelView.detect_zone(model_pos):
                    case Zone.BODY:
                        # just heat up
                        self._rezonator_model.heat_body(
                            cmd_s, self._modeling_period)
                    case Zone.FORBIDDEN:
                        # heat up and add energy to forbidden zone
                        self._rezonator_model.heat_forbidden(
                            cmd_s, self._modeling_period)
                    case Zone.TARGET1 | Zone.TARGET2 as zone:
                        # Обработка мишеней
                        pos = ModelView.map_to_zone(zone, model_pos)
                        self._rezonator_model.target(
                            zone, pos, cmd_s, self._modeling_period)
                    case _:
                        self._rezonator_model.idle(self._modeling_period)

            self._period_accum += ts

            self._shift_move_history(
                command['destination'], S=cmd_s, F=command['speed'])

    def _shift_move_history(self, dest_pos: tuple[float, float], S: float, F: float):
        self._move_history.dequeue()
        self._move_history.enqueue(np.array([[*dest_pos, S, F]]))

    # def _map_adj_to_relatie(self, adj: float) -> float:
    #    """
    #    Преобразование изменения частоты резонатора в относительное значение приведенное к максимальному теоритическому изменинию
    #    :param adj: Абсолютное значение изменения частоты
    #    :return: Относительное значение
    #    """
    #    return adj / self._rezonator_model.possible_freq_adjust

    # def tick(self, cycle_time: float, model_pos) -> dict:
    #    """
    #    Шаг симуляции
    #    :param time: Время шага
    #    """
#
    #    current_pos = self._work_zone.map_from_global(
    #        self._movement.current_position)
    #    current_target = self._work_zone.map_from_global(
    #        self._movement.move_target)
    #    current_s = self._work_zone.map_s_from_global(
    #        self._movement.current_s)
    #    current_f = self._work_zone.map_f_from_global(
    #        self._movement.current_f)
#
    #    match ModelView.detect_zone(model_pos):
    #        case Zone.BODY:
    #            # just heat up
    #            self._rezonator_model.heat_body(current_s, cycle_time)
    #        case Zone.FORBIDDEN:
    #            # heat up and add energy to forbidden zone
    #            self._rezonator_model.heat_forbidden(current_s, cycle_time)
    #        case Zone.TARGET1 | Zone.TARGET2 as zone:
    #            # Обработка мишеней
    #            pos = ModelView.map_to_zone(zone, model_pos)
    #            self._rezonator_model.target(
    #                zone, pos, current_s, cycle_time)
    #        case _:
    #            self._rezonator_model.idle(cycle_time)
#
    #    passed_path_len = self._movement.tick(cycle_time)
#
    #    result = self._controller.update({
    #        'current_pos': current_pos,
    #        'target_pos': current_target,
    #        'current_s': current_s,
    #        'current_f': current_f,
    #        'freq_history': self._measure_diff_history,
    #    })
#
    #    destination = result['destination']
    #    #print("({}, {}) -> ({}, {})".format(current_pos[0], current_pos[1], destination[0], destination[1]))
#
    #    self._period_accum += cycle_time
    #    if self._period_accum > self._freqmeter_period:
    #        self._period_accum -= self._freqmeter_period
#
    #        current_freq = self._rezonator_model.get_metrics()['freq_change']
    #        self._measure_diff_history.append(
    #            self._initial_freq_diff - self._map_adj_to_relatie(current_freq))
    #        self._measure_diff_history.pop(0)
#
    #    # try send new command
    #    cmd = Command(
    #        destinanton=self._work_zone.map_to_global(destination),
    #        F=self._work_zone.map_f_to_global(result['speed']),
    #        S=self._work_zone.map_s_to_global(result['power']))
    #    # print(cmd)
    #    self._movement.process(cmd)
#
    #    return {
    #        'F': current_f,
    #        'S': current_s,
    #        'self_grade': result['self_grade'],
    #        'Passed': self._work_zone.map_path_len_from_global(passed_path_len),
    #    }

    # def use_rezonator(self, f, *args, **kwargs):
    #    """
    #    Применить функцию к резонатору, сохраняя его состояние
    #    """
    #    return f(self._rezonator_model, *args, **kwargs)

    # def laser_pos(self) -> tuple[float, float]:
    #    """
    #    Функция возвращает текущую глобальную позицию лазера
    #    """
    #    relative_pos = self._work_zone.map_from_global(
    #        self._movement.current_position)
    #    local_pos = self._work_zone.map_relative_to_local(relative_pos)
    #    return self._rezonator_model.map_local_to_global(local_pos)

    # def laser_rel_power(self) -> float:
    #    """
    #    Текущая отностительная мощность лазера [0..1]
    #    """
    #    return self._work_zone.map_s_from_global(self._movement.current_s)
