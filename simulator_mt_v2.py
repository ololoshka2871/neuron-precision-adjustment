#!/usr/bin/env python

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from misc.common import draw_polygon, gen_sigmoid
from adjust_zone_model import draw_model
from controllers.manual_controller import ManualController
#from controller_grader import ControllerGrager
from models.rezonator_model import RezonatorModel, Playground, ModelView
#from sim_stop_detector import SimStopDetector, StopCondition
from simulators.simulator_v2 import Simulator


class ControllerInputDisplay:
    def __init__(self, pg_ax: Axes, model_ax: Axes, info_ax: Axes, 
                 playground: Playground, model_view: ModelView, move_history_size: int):

        # рисуем базовую форму
        draw_polygon(pg_ax, playground.rezonator,
                    edgecolor='black', facecolor='none')

        # рисуем цели
        draw_polygon(pg_ax, playground.target(0), color='green')
        draw_polygon(pg_ax, playground.target(1), color='green')

        # рисуем запрещенную область
        draw_polygon(pg_ax, playground.forbidden_area, color='magenta')

        # рисуем рабочую область
        draw_polygon(pg_ax, playground.working_area,
                    edgecolor='blue', facecolor='none')

        # текущая точка
        self.current_pos_pg = pg_ax.plot([0], [0], 'ro')

        # ----------------------------------------

        # рисуем базовую форму
        draw_polygon(model_ax, model_view.rezonator,
                    edgecolor='black', facecolor='none')

        # рисуем цели
        self_patches = [draw_model(model_ax, model_view.target(i)) for i in range(2)]

        # рисуем запрещенную область
        draw_polygon(model_ax, model_view.forbidden_area, color='magenta')

        # рисуем рабочую область
        draw_polygon(model_ax, model_view.working_area,
                    edgecolor='blue', facecolor='none')

        # текущая точка
        #model_pos = playground.map_to_model(current_pos_global)
        #current_pos_transformed, = mv.plot(*model_pos, 'ro')

        # Установка границ по осям X и Y чтобы видно было только рабочую область
        limits = model_view.working_area_limits(0.1)
        model_ax.set_xlim(*limits[0])
        model_ax.set_ylim(*limits[1])

        # ----------------------------------------

        # Установка границ по осям X и Y чтобы видно было только рабочую область
        #limits = playground.working_area_limits(0.1)
        #pg.set_xlim(*limits[0])
        #pg.set_ylim(*limits[1])
#
        #playground.set_xlim(-1, 1)
        #playground.set_ylim(-1, 1)
        #self._tail = [playground.plot([0], [0], 'o-')[0]
        #              for _ in range(move_history_size)]
#
        #self._freq_history, = info.plot([0], [0], '-g')
        #info.set_xlim(0, 1)
        #info.set_ylim(0, 1)
        #self._info = info

    def __call__(self, input: dict) -> None:
        #start: tuple[float, float] = input['move_history'][0][:2]
#
        #for curve, move_history_item in zip(self._tail, input['move_history']):
        #    dest_x, dest_y, S, F = move_history_item
        #    curve.set_data([start[0], dest_x], [start[1], dest_y])
        #    curve.set_color((F, 0, 0, S))  # type: ignore
        #    start = move_history_item[:2]
#
        #freq_history = input['freq_history']
#
        #x = np.linspace(0.0, 1.0, len(freq_history))
        #self._freq_history.set_data(x, freq_history)
        #self._info.relim()
        pass


if __name__ == "__main__":
    LASER_POWER = 30.0  # [W]
    F_HISTORY_SIZE = 10
    MOVE_HISTORY_SIZE = 10
    POWER_THRESHOLD = 0.05
    DEST_FREQ_CH = 50.0
    MAX_T = 100.0

    SIM_CYCLE_TIME = 0.01
    SIM_TIMEOUT = 10.0

    f, ax = plt.subplots(1, 3)

    rezonator = RezonatorModel(power_threshold=POWER_THRESHOLD)

    # Генерируем случайное смещение и случайный угол поворота
    offset = (np.random.random() * 0.3, np.random.random() * 0.5)
    angle = np.random.random() * 20 - 10
    print('offset: {}, angle: {}'.format(offset, angle))

    playground = rezonator.get_playground(offset, angle)
    model_view = rezonator.get_model_view(offset, angle)

    # NNController.init_model(F_HISTORY_SIZE)

    # weights = NNController.shuffled_weights()

    sim = Simulator(rezonator, ManualController(), (0, 0),
                    freq_history_size=F_HISTORY_SIZE)

    # sim_stop_detector = SimStopDetector(timeout=SIM_TIMEOUT,
    #                                    history_len_s=1.0,
    #                                    min_path=0.01,
    #                                    min_avg_speed=0.05,
    #                                    min_laser_power=POWER_THRESHOLD * 0.5,
    #                                    max_temperature=MAX_T)

    # grader = ControllerGrager(dest_freq_ch=DEST_FREQ_CH,
    #                          f_penalty=gen_sigmoid(
    #                              k=1.0 / LASER_POWER, x_offset_to_right=-6),
    #                          max_temperature=MAX_T)

    # ----------------------------------------
#
    #now = 0
#
    # ----------------------------------------
#
    #rmetrics = sim.use_rezonator(RezonatorModel.get_metrics)
#
    #mp = ax[2]
#
    # рисуем температуру
    # tc, = mp.plot(dt.datetime.fromtimestamp(
    #    now), rmetrics['temperature'], 'r-')
#
    # рисуем изменение частоты
    # fс, = mp.plot(dt.datetime.fromtimestamp(
    #    now), rmetrics['freq_change'], 'b-')
#
    # Рисуем диссбаланс
    # dc, = mp.plot(dt.datetime.fromtimestamp(
    #    now), rmetrics['disbalance'], ':')
#
    # форматирование оси X
    # устанавливаем интервал в 1 секунду
    #hours = mdates.SecondLocator(interval=1)
    # time_format = mdates.DateFormatter('%S')  # устанавливаем формат времени
    # mp.xaxis.set_major_locator(hours)  # устанавливаем локатор основных делений
    # устанавливаем форматтер основных делений
    # mp.xaxis.set_major_formatter(time_format)
#
    # ----------------------------------------

    input_display = ControllerInputDisplay(
        *ax, playground, model_view, MOVE_HISTORY_SIZE  # type: ignore
    )

    plt.show(block=False)

    sim.perform_modeling(playground, 0, input_display)

    # while True:
    #    pos = sim.laser_pos()
    #    current_pos.set_data(pos)
#
    #    model_pos = playground.map_to_model(pos)
    #    model_power = sim.laser_rel_power()
    #    current_pos_transformed.set_data(model_pos)
    #    current_pos_transformed.set_markerfacecolor(
    #        (1.0, 0.0, 0.0, model_power))  # type: ignore
#
    #    mmetrics = sim.tick(SIM_CYCLE_TIME, model_pos)
    #    rmetrics = sim.use_rezonator(RezonatorModel.get_metrics)
#
    #    model_view = sim.use_rezonator(
    #        RezonatorModel.get_model_view, offset, angle)
#
    #    for target in range(2):
    #        target_colors = model_view.target_color_map(target)
    #        for row in zip(patches[target], target_colors):
    #            for patch, color in zip(*row):
    #                patch.set_facecolor(color)
#
    #    # ------------ Условие останова ----------
#
    #    stop_condition = sim_stop_detector.tick(
    #        SIM_CYCLE_TIME, mmetrics, rmetrics)
#
    #    if stop_condition != StopCondition.NONE:
    #        grade = grader.get_grade(
    #            rmetrics, sim_stop_detector.summary(), stop_condition)
    #        print(
    #            f"Done {stop_condition}; Fd:{grade[0]:.2f}, db:{grade[1]:.2f}, pen:{grade[2]:.2f}, t:{grade[3]:.2f}, ss:{grade[4]:.2f}, Tmax:{grade[5]:.2f}, Va:{grade[6]:.2f}")
    #        sf, ax = plt.subplots(1, 1)
    #        sim_stop_detector.plot_summary(ax)
    #        plt.show(block=True)
    #        break
#
    #    # ------------ Метрики -------------------
#
    #    d = tc.get_data(orig=True)
#
    #    points = min(100, len(d[0]))
    #    ts = dt.datetime.fromtimestamp(now)
#
    #    d = [list(d[0][-points:]), list(d[1][-points:])]
    #    d[0].append(ts)  # type: ignore
    #    d[1].append(rmetrics['temperature'])
    #    tc.set_data(d)
#
    #    d = fс.get_data(orig=True)
    #    d = [list(d[0][-points:]), list(d[1][-points:])]
    #    d[0].append(ts)  # type: ignore
    #    d[1].append(rmetrics['freq_change'])
    #    fс.set_data(d)
#
    #    d = dc.get_data(orig=True)
    #    d = [list(d[0][-points:]), list(d[1][-points:])]
    #    d[0].append(ts)  # type: ignore
    #    d[1].append(rmetrics['disbalance'] * 100)
    #    dc.set_data(d)
#
    #    mp.relim()
    #    mp.autoscale_view()
#
    #    print(
    #        f"Static freq change: {rmetrics['static_freq_change']:.2f} Hz, disbalance: {rmetrics['disbalance'] * 100:.2f} %")
#
    #    # ----------------------------------------
#
    #    now += SIM_CYCLE_TIME
#
    #    plt.draw()
    #    plt.pause(0.0001)
