#!/usr/bin/env python

import time

import numpy as np

import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from common import draw_polygon, gen_sigmoid
from adjust_zone_model import draw_model
from controller import NNController
from controller_grader import ControllerGrager
from rezonator_model import RezonatorModel
from sim_stop_detector import SimStopDetector, StopCondition
from simulator import Simulator


if __name__ == "__main__":
    LASER_POWER = 30.0  # [W]
    HISTORY_SIZE = 10
    POWER_THRESHOLD = 0.05
    DEST_FREQ_CH = 50.0
    MAX_T = 100.0

    f, ax = plt.subplots(1, 3)

    NNController.init_model(HISTORY_SIZE)

    weights = NNController.shuffled_weights()

    sim = Simulator(RezonatorModel(power_threshold=POWER_THRESHOLD),
                    NNController(weights), (-100, 15),
                    freq_history_size=HISTORY_SIZE)

    sim_stop_detector = SimStopDetector(timeout=10.0,
                                        history_len_s=1.0,
                                        min_path=0.01,
                                        min_avg_speed=0.05,
                                        min_laser_power=POWER_THRESHOLD * 0.5,
                                        max_temperature=MAX_T)

    grader = ControllerGrager(dest_freq_ch=DEST_FREQ_CH,
                              f_penalty=gen_sigmoid(
                                  k=1.0 / LASER_POWER, x_offset_to_right=-6),
                              max_temperature=MAX_T)

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

        stop_condition = sim_stop_detector.tick(cycle_time, mmetrics, rmetrics)

        if stop_condition != StopCondition.NONE:
            grade = grader.get_grade(
                rmetrics, sim_stop_detector.summary(), stop_condition)
            print(
                f"Done {stop_condition}; Fd:{grade[0]:.2f}, db:{grade[1]:.2f}, pen:{grade[2]:.2f}, t:{grade[3]:.2f}, ss:{grade[4]:.2f}, Tmax:{grade[5]:.2f}, Va:{grade[6]:.2f}")
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
