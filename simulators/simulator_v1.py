from rezonator_model import RezonatorModel, Zone, ModelView
from moving_interpolator import MovingInterpolator, Command
from misc.work_zone import WorkZone, Rect


class Simulator:
    """
    Симуляция обработки лазером резонатора
    - Резонатор представлен классом Resonator
    - Станок симулирует класс MovingInterpolator
    - Класс WorkZone преобразует коориднаты между Станком и сонтроллером
    - Контроллер получает информацию о резонаторе и станке и выдает новые команды станку
    """

    def __init__(self, rezonator_model: RezonatorModel, controller,
                 global_center: tuple[float, float],
                 initial_freq_diff=0.5,
                 max_f=1000.0,
                 freqmeter_period=0.4,
                 laser_power_max=255.0,
                 freq_history_size=10):
        """
        :param rezonator: Резонатор
        :param controller: Контроллер
        :param global_center: Координаты центра рабочей зоны в глобальной системе координат
        :param initial_freq_diff: Для от максимально-возможной перестройки частоты для этой симуляции
        :param max_f: Максимальная скорость перемещения станка
        :param freqmeter_period: Период измерения частоты, [с]
        :param laser_power_max: Максимальная мощность лазера [W]
        """
        self._rezonator_model = rezonator_model
        self._controller = controller
        self._freqmeter_period = freqmeter_period

        self._work_zone = WorkZone(Rect.from_rezonator(
            RezonatorModel.REZONATOR, global_center),
            laser_power_max, max_f)

        self._period_accum = 0.0

        self._moving_interpolator = MovingInterpolator()
        self._moving_interpolator.begin(
            self._work_zone.map_to_global((0, 1.0)))

        self._initial_freq_diff = initial_freq_diff
        self._measure_diff_history = [
            initial_freq_diff for _ in range(freq_history_size)]

    def _map_adj_to_relatie(self, adj: float) -> float:
        """
        Преобразование изменения частоты резонатора в относительное значение приведенное к максимальному теоритическому изменинию
        :param adj: Абсолютное значение изменения частоты
        :return: Относительное значение
        """
        return adj / self._rezonator_model.possible_freq_adjust

    def tick(self, cycle_time: float, model_pos) -> dict:
        """
        Шаг симуляции
        :param time: Время шага
        """

        current_pos = self._work_zone.map_from_global(
            self._moving_interpolator.current_position)
        current_target = self._work_zone.map_from_global(
            self._moving_interpolator.move_target)
        current_s = self._work_zone.map_s_from_global(
            self._moving_interpolator.current_s)
        current_f = self._work_zone.map_f_from_global(
            self._moving_interpolator.current_f)

        match ModelView.detect_zone(model_pos):
            case Zone.BODY:
                # just heat up
                self._rezonator_model.heat_body(current_s, cycle_time)
            case Zone.FORBIDDEN:
                # heat up and add energy to forbidden zone
                self._rezonator_model.heat_forbidden(current_s, cycle_time)
            case Zone.TARGET1 | Zone.TARGET2 as zone:
                # Обработка мишеней
                pos = ModelView.map_to_zone(zone, model_pos)
                self._rezonator_model.target(
                    zone, pos, current_s, cycle_time)
            case _:
                self._rezonator_model.idle(cycle_time)

        passed_path_len = self._moving_interpolator.tick(cycle_time)

        result = self._controller.update({
            'current_pos': current_pos,
            'target_pos': current_target,
            'current_s': current_s,
            'current_f': current_f,
            'freq_history': self._measure_diff_history,
        })

        destination = result['destination']
        #print("({}, {}) -> ({}, {})".format(current_pos[0], current_pos[1], destination[0], destination[1]))

        self._period_accum += cycle_time
        if self._period_accum > self._freqmeter_period:
            self._period_accum -= self._freqmeter_period

            current_freq = self._rezonator_model.get_metrics()['freq_change']
            self._measure_diff_history.append(
                self._initial_freq_diff - self._map_adj_to_relatie(current_freq))
            self._measure_diff_history.pop(0)

        # try send new command
        cmd = Command(
            destinanton=self._work_zone.map_to_global(destination),
            F=self._work_zone.map_f_to_global(result['speed']),
            S=self._work_zone.map_s_to_global(result['power']))
        # print(cmd)
        self._moving_interpolator.process(cmd)

        return {
            'F': current_f,
            'S': current_s,
            'self_grade': result['self_grade'],
            'Passed': self._work_zone.map_path_len_from_global(passed_path_len),
        }

    def use_rezonator(self, f, *args, **kwargs):
        """
        Применить функцию к резонатору, сохраняя его состояние
        """
        return f(self._rezonator_model, *args, **kwargs)

    def laser_pos(self) -> tuple[float, float]:
        """
        Функция возвращает текущую глобальную позицию лазера
        """
        relative_pos = self._work_zone.map_from_global(
            self._moving_interpolator.current_position)
        local_pos = self._work_zone.map_relative_to_local(relative_pos)
        return self._rezonator_model.map_local_to_global(local_pos)

    def laser_rel_power(self) -> float:
        """
        Текущая отностительная мощность лазера [0..1]
        """
        return self._work_zone.map_s_from_global(self._moving_interpolator.current_s)

    def _generate_test_polygon_local(self) -> list:
        pos = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]

        local_positions = [
            self._work_zone.map_relative_to_local(p) for p in pos]
        abs_psitions = [self._rezonator_model.map_local_to_global(
            p) for p in local_positions]
        return abs_psitions

    def _generate_test_polygon_global(self) -> list:
        pos = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]

        gp = [self._work_zone.map_to_global(p) for p in pos]
        lp = [self._work_zone.map_from_global(p) for p in gp]
        return lp
