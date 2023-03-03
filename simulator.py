#!/usr/bin/env python

import time

from rezonator_model import RezonatorModel, Zone
from moving_interpolator import MovingInterpolator, Command
from work_zone import WorkZone, Rect
from controller import RandomController


class Simulator:
    """
    Симуляция обработки лазером резонатора
    - Резонатор представлен классом Resonator
    - Станок симулирует класс MovingInterpolator
    - Класс WorkZone преобразует коориднаты между Станком и сонтроллером
    - Контроллер получает информацию о резонаторе и станке и выдает новые команды станку
    """

    FREQ_DIFF_HISTORY_SIZE = 10

    def __init__(self, rezonator_model: RezonatorModel, controller,
                 global_center: tuple[float, float],
                 initial_freq_diff=0.5,
                 max_f=1000.0,
                 freqmeter_period=0.4,
                 laser_power_max=255.0):
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

        self._moving_interpolator = MovingInterpolator()
        self._moving_interpolator.begin(
            self._work_zone.map_to_global((0, 1.0)), time.time())

        self._initial_freq_diff = initial_freq_diff
        self._measure_diff_history = [
            initial_freq_diff for _ in range(Simulator.FREQ_DIFF_HISTORY_SIZE)]
        self._now = time.time()

    def _map_adj_to_relatie(self, adj: float) -> float:
        """
        Преобразование изменения частоты резонатора в относительное значение приведенное к максимальному теоритическому изменинию
        :param adj: Абсолютное значение изменения частоты
        :return: Относительное значение
        """
        return adj / self._rezonator_model.possible_freq_adjust

    def tick(self, time: float):
        """
        Шаг симуляции
        :param time: Время шага
        """
        self._moving_interpolator.tick(time)

        current_pos = self._work_zone.map_from_global(
            self._moving_interpolator.current_position())
        current_s = self._work_zone.map_s_from_global(
            self._moving_interpolator.current_s)
        current_f = self._work_zone.map_f_from_global(
            self._moving_interpolator.current_f)

        if time - self._now > self._freqmeter_period:
            self._now = time
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

        # try send new command
        self._moving_interpolator.process(Command(
            self._work_zone.map_to_global(result['destination']),
            self._work_zone.map_s_to_global(result['power']),
            self._work_zone.map_f_to_global(result['speed']))
        )

    def use_rezonator(self, f, *args, **kwargs):
        """
        Применить функцию к резонатору, сохраняя его состояние
        """
        return f(self._rezonator_model, *args, **kwargs)

    def laser_global_pos(self) -> tuple[float, float]:
        """
        Функция возвращает текущую глобальную позицию лазера
        """
        relative_pos = self._work_zone.map_from_global(
            self._moving_interpolator.current_position())
        local_pos = self._work_zone.map_relative_to_local(relative_pos)
        return self._rezonator_model.map_local_to_global(local_pos)


if __name__ == "__main__":
    import time

    import numpy as np

    import datetime as dt
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from common import draw_polygon
    from adjust_zone_model import draw_model

    LASER_POWER = 30.0  # [W]

    f, ax = plt.subplots(1, 3)

    sim = Simulator(RezonatorModel(),
                    RandomController(), (-100, 15))

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

    # текущая точка
    current_pos_global = sim.laser_global_pos()
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

    m = sim.use_rezonator(RezonatorModel.get_metrics)

    mp = ax[2]

    # рисуем температуру
    tc, = mp.plot(dt.datetime.fromtimestamp(start), m['temperature'], 'r-')

    # рисуем изменение частоты
    fс, = mp.plot(dt.datetime.fromtimestamp(start), m['freq_change'], 'b-')

    # Рисуем диссбаланс
    dc, = mp.plot(dt.datetime.fromtimestamp(start), m['disbalance'], ':')

    # форматирование оси X
    # устанавливаем интервал в 1 секунду
    hours = mdates.SecondLocator(interval=1)
    time_format = mdates.DateFormatter('%S')  # устанавливаем формат времени
    mp.xaxis.set_major_locator(hours)  # устанавливаем локатор основных делений
    # устанавливаем форматтер основных делений
    mp.xaxis.set_major_formatter(time_format)

    # ----------------------------------------

    plt.show(block=True)

    cycle_time = 0

    # while True:
    #    click = f.ginput(show_clicks=False, timeout=0.01)
    #    if len(click) != 0:
    #        click = click[0]
#
    #        current_pos.set_data(click[0], click[1])
#
    #        model_pos = playground.map_to_model(click)
    #        current_pos_transformed.set_data(model_pos[0], model_pos[1])
#
    #        match model_view.detect_zone(model_pos):
    #            case Zone.BODY:
    #                # just heat up
    #                rezonator.heat_body(LASER_POWER, cycle_time)
    #            case Zone.FORBIDDEN:
    #                # heat up and add energy to forbidden zone
    #                rezonator.heat_forbidden(LASER_POWER, cycle_time)
    #            case Zone.TARGET1 | Zone.TARGET2 as zone:
    #                # Обработка мишеней
    #                pos = model_view.map_to_zone(zone, model_pos)
    #                rezonator.target(zone, pos, LASER_POWER, cycle_time)
#
    #                # Обновление цвета мишеней
    #                # Генерируем новую модель по результатам обновления
    #                model_view = rezonator.get_model_view(offset, angle)
    #                for i in range(2):
    #                    target_colors = model_view.target_color_map(i)
    #                    for row in zip(patches[i], target_colors):
    #                        for patch, color in zip(*row):
    #                            patch.set_facecolor(color)
#
    #    else:
    #        rezonator.idle(cycle_time)
#
    #    # ------------ Метрики -------------------
#
    #    m = rezonator.get_metrics()
#
    #    d = tc.get_data(orig=True)
#
    #    points = min(100, len(d[0]))
    #    ts = dt.datetime.fromtimestamp(time.time())
#
    #    d = [list(d[0][-points:]), list(d[1][-points:])]
    #    d[0].append(ts)  # type: ignore
    #    d[1].append(m['temperature'])
    #    tc.set_data(d)
#
    #    d = fс.get_data(orig=True)
    #    d = [list(d[0][-points:]), list(d[1][-points:])]
    #    d[0].append(ts)  # type: ignore
    #    d[1].append(m['freq_change'])
    #    fс.set_data(d)
#
    #    d = dc.get_data(orig=True)
    #    d = [list(d[0][-points:]), list(d[1][-points:])]
    #    d[0].append(ts)  # type: ignore
    #    d[1].append(m['disbalance'] * 100)
    #    dc.set_data(d)
#
    #    mp.relim()
    #    mp.autoscale_view()
#
    #    print(
    #        f"Static freq change: {m['static_freq_change']:.2f} Hz, disbalance: {m['disbalance'] * 100:.2f} %")
#
    #    # ----------------------------------------
#
    #    cycle_time = time.time() - start
    #    start = time.time()
#
    #    plt.draw()
#
