#!/usr/bin/env python

import time

from rezonator_model import RezonatorModel, Zone, ModelView
from moving_interpolator import MovingInterpolator, Command
from work_zone import WorkZone, Rect
from controller import NNController
from sim_stop_detector import SimStopDetector, StopCondition
from controller_grader import ControllerGrager


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
        self._moving_interpolator.tick(cycle_time)

        current_pos = self._work_zone.map_from_global(
            self._moving_interpolator.current_position)
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

        current_freq = self._rezonator_model.get_metrics()['freq_change']

        self._measure_diff_history.append(
            self._map_adj_to_relatie(current_freq))
        self._measure_diff_history.pop(0)

        result = self._controller.update({
            'current_pos': current_pos,
            'current_s': current_s,
            'current_f': current_f,
            'freq_history': self._measure_diff_history,
        })

        self._period_accum += cycle_time
        if self._period_accum > self._freqmeter_period:
            self._period_accum -= self._freqmeter_period

            # try send new command
            cmd = Command(
                destinanton=self._work_zone.map_to_global(
                    result['destination']),
                F=self._work_zone.map_f_to_global(result['speed']),
                S=self._work_zone.map_s_to_global(result['power']))
            # print(cmd)
            self._moving_interpolator.process(cmd)

        return {
            'F': current_f,
            'S': current_s,
            'T': self._rezonator_model.current_temperature_K(),
            'self_grade': result['self_grade']
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


if __name__ == "__main__":
    import time

    import numpy as np

    import datetime as dt
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from common import draw_polygon, gen_sigmoid
    from adjust_zone_model import draw_model

    LASER_POWER = 30.0  # [W]
    HISTORY_SIZE = 10
    POWER_THRESHOLD = 0.05
    DEST_FREQ_CH = 50.0
    MAX_T = 100.0

    def grade_stop_condition(sc: StopCondition) -> float:
        match sc:
            case StopCondition.TIMEOUT:
                return -0.5
            case  StopCondition.STALL:
                return -0.7
            case  StopCondition.LOW_POWER:
                return -0.6
            case  StopCondition.OVERHEAT:
                return -0.2
            case  StopCondition.SELF_STOP:
                return 0.2
            case _:
                return 0.0

    f, ax = plt.subplots(1, 3)

    NNController.init_model(HISTORY_SIZE)

    sim = Simulator(RezonatorModel(power_threshold=POWER_THRESHOLD),
                    NNController(), (-100, 15),
                    freq_history_size=HISTORY_SIZE)

    sim_stop_detector = SimStopDetector(timeout=10.0,
                                        history_len_s=5.0,
                                        min_avg_speed=0.05,
                                        min_laser_power=POWER_THRESHOLD * 0.5,
                                        max_temperature=MAX_T)

    grader = ControllerGrager(dest_freq_ch=DEST_FREQ_CH,
                              f_penalty=gen_sigmoid(
                                  k=1.0 / LASER_POWER, x_offset=-6),
                              max_temperature=MAX_T,
                              grade_stop_condition=grade_stop_condition)

    # Генерируем случайное смещение и случайный угол поворота
    offset = (np.random.random() * 0.3, np.random.random() * 0.5)
    angle = np.random.random() * 20 - 10
    print('offset: {}, angle: {}'.format(offset, angle))

    # ----------------------------------------

    pg = ax[0]

    playground = sim.use_rezonator(
        RezonatorModel.get_playground, offset, angle)

    # рисуем базовую форму
    draw_polygon(pg, playground.rezonator,
                 edgecolor='black', facecolor='none')

    # рисуем цели
    draw_polygon(pg, playground.target(0), color='green')
    draw_polygon(pg, playground.target(1), color='green')

    # рисуем запрещенную область
    draw_polygon(pg, playground.forbidden_area, color='magenta')

    # рисуем рабочую область
    draw_polygon(pg, playground.working_area,
                 edgecolor='blue', facecolor='none')

    # test field
    # pg.add_patch(Polygon(sim.generate_test_polygon_local(), facecolor=(0,0,0,0.3)))

    # текущая точка
    current_pos_global = sim.laser_pos()
    current_pos, = pg.plot(*current_pos_global, 'ro')

    # Установка границ по осям X и Y чтобы видно было только рабочую область
    limits = playground.working_area_limits(0.1)
    pg.set_xlim(*limits[0])
    pg.set_ylim(*limits[1])

    # ----------------------------------------

    mv = ax[1]

    model_view = sim.use_rezonator(
        RezonatorModel.get_model_view, offset, angle)

    # рисуем базовую форму
    draw_polygon(mv, model_view.rezonator,
                 edgecolor='black', facecolor='none')

    # рисуем цели
    patches = [draw_model(mv, model_view.target(i)) for i in range(2)]

    # рисуем запрещенную область
    draw_polygon(mv, model_view.forbidden_area, color='magenta')

    # рисуем рабочую область
    draw_polygon(mv, model_view.working_area,
                 edgecolor='blue', facecolor='none')

    # текущая точка
    model_pos = playground.map_to_model(current_pos_global)
    current_pos_transformed, = mv.plot(*model_pos, 'ro')

    # Установка границ по осям X и Y чтобы видно было только рабочую область
    limits = model_view.working_area_limits(0.1)
    mv.set_xlim(*limits[0])
    mv.set_ylim(*limits[1])

    # ----------------------------------------

    start = time.time()

    # ----------------------------------------

    rmetrics = sim.use_rezonator(RezonatorModel.get_metrics)

    mp = ax[2]

    # рисуем температуру
    tc, = mp.plot(dt.datetime.fromtimestamp(
        start), rmetrics['temperature'], 'r-')

    # рисуем изменение частоты
    fс, = mp.plot(dt.datetime.fromtimestamp(
        start), rmetrics['freq_change'], 'b-')

    # Рисуем диссбаланс
    dc, = mp.plot(dt.datetime.fromtimestamp(
        start), rmetrics['disbalance'], ':')

    # форматирование оси X
    # устанавливаем интервал в 1 секунду
    hours = mdates.SecondLocator(interval=1)
    time_format = mdates.DateFormatter('%S')  # устанавливаем формат времени
    mp.xaxis.set_major_locator(hours)  # устанавливаем локатор основных делений
    # устанавливаем форматтер основных делений
    mp.xaxis.set_major_formatter(time_format)

    # ----------------------------------------

    plt.show(block=False)

    cycle_time = 0

    while True:
        pos = sim.laser_pos()
        current_pos.set_data(pos)

        model_pos = playground.map_to_model(pos)
        model_power = sim.laser_rel_power()
        current_pos_transformed.set_data(model_pos)
        current_pos_transformed.set_markerfacecolor(
            (1.0, 0.0, 0.0, model_power))  # type: ignore

        mmetrics = sim.tick(cycle_time, model_pos)
        rmetrics = sim.use_rezonator(RezonatorModel.get_metrics)

        model_view = sim.use_rezonator(
            RezonatorModel.get_model_view, offset, angle)

        for target in range(2):
            target_colors = model_view.target_color_map(target)
            for row in zip(patches[target], target_colors):
                for patch, color in zip(*row):
                    patch.set_facecolor(color)

        # ------------ Условие останова ----------

        stop_condition = sim_stop_detector.tick(cycle_time, mmetrics)

        if stop_condition != StopCondition.NONE:
            g = grader.get_grade(
                rmetrics, sim_stop_detector.summary(), stop_condition)
            print(
                f"Simulation stop detected: {stop_condition}, sim_result = {g}")
            sf, ax = plt.subplots(1, 1)
            sim_stop_detector.plot_summary(ax)
            plt.show(block=True)
            break

        # ------------ Метрики -------------------

        d = tc.get_data(orig=True)

        points = min(100, len(d[0]))
        ts = dt.datetime.fromtimestamp(time.time())

        d = [list(d[0][-points:]), list(d[1][-points:])]
        d[0].append(ts)  # type: ignore
        d[1].append(rmetrics['temperature'])
        tc.set_data(d)

        d = fс.get_data(orig=True)
        d = [list(d[0][-points:]), list(d[1][-points:])]
        d[0].append(ts)  # type: ignore
        d[1].append(rmetrics['freq_change'])
        fс.set_data(d)

        d = dc.get_data(orig=True)
        d = [list(d[0][-points:]), list(d[1][-points:])]
        d[0].append(ts)  # type: ignore
        d[1].append(rmetrics['disbalance'] * 100)
        dc.set_data(d)

        mp.relim()
        mp.autoscale_view()

        print(
            f"Static freq change: {rmetrics['static_freq_change']:.2f} Hz, disbalance: {rmetrics['disbalance'] * 100:.2f} %")

        # ----------------------------------------

        cycle_time = time.time() - start
        start = time.time()

        plt.draw()
        plt.pause(0.0001)
