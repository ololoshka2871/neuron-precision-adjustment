#!/usr/bin/env python

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from constants_v2 import *
from misc.Rezonator import Rezonator
from misc.common import draw_polygon, gen_sigmoid, create_tail
from models.adjust_zone_model import draw_model
from controllers.manual_controller import ManualController
from misc.coordinate_transformer import CoordinateTransformer, WorkzoneRelativeCoordinates
from misc.f_s_transformer import FSTransformer
from graders.controller_grader_v2 import ControllerGrager
from models.rezonator_model import RezonatorModel, ModelView
from models.sim_stop_detector_v2 import SimStopDetector
from simulators.simulator_v2 import Simulator
from controllers.controller_v2 import NNController


class ControllerInputDisplay:
    def __init__(self, wz_ax: Axes, model_ax: Axes, info_ax: Axes,
                 rez: Rezonator, model_view: ModelView, ct: CoordinateTransformer,
                 move_history_size: int, initial_pos: WorkzoneRelativeCoordinates):
        self._ct = ct
        self._model_view = model_view

        # базовая точка - середина в месте крепления (0, 0)
        rezonator = rez['rezonator']

        # первая ветка
        target1 = rez['targets'][0]

        # вторая ветка
        target2 = rez['targets'][1]

        # рабочая область
        working_area = rez['working_area']

        # Запрещенная область
        forbidden_area = rez['forbidden_area']

        # ----------------------------------------

        real_rezonator = ct.array_wrap_from_model_to_real(rezonator)
        real_target1 = ct.array_wrap_from_model_to_real(target1)
        real_target2 = ct.array_wrap_from_model_to_real(target2)
        real_forbidden_area = ct.array_wrap_from_model_to_real(forbidden_area)
        real_working_area = ct.get_real_working_zone(working_area)

        # ----------------------------------------

        wz_ax.set_title('Рабочая зона')

        wz_rezonator = ct.array_wrap_from_real_to_workzone(real_rezonator)
        wz_target1 = ct.array_wrap_from_real_to_workzone(real_target1)
        wz_target2 = ct.array_wrap_from_real_to_workzone(real_target2)
        wz_forbidden_area = ct.array_wrap_from_real_to_workzone(
            real_forbidden_area)
        wz_working_area = ct.array_wrap_from_real_to_workzone(
            real_working_area)

        # рисуем базовую форму
        draw_polygon(wz_ax, wz_rezonator, edgecolor='black', facecolor='none')

        # рисуем цели
        draw_polygon(wz_ax, wz_target1, color='green')
        draw_polygon(wz_ax, wz_target2, color='green')

        # рисуем запрещенную область
        draw_polygon(wz_ax, wz_forbidden_area, color='magenta')

        # рисуем рабочую область
        draw_polygon(wz_ax, wz_working_area,
                     edgecolor='blue', facecolor='none')

        self._wz_tail = create_tail(
            wz_ax, move_history_size, initial_pos.tuple())

        # Установка границ по осям X и Y чтобы видно было только рабочую область
        wz_ax.set_xlim(-1.0, 1.0)
        wz_ax.set_ylim(-1.0, 1.0)

        # ----------------------------------------

        model_ax.set_title('Модель')

        model_rezonator = ct.array_wrap_from_real_to_model(real_rezonator)
        model_forbidden_area = ct.array_wrap_from_real_to_model(
            real_forbidden_area)
        model_working_area = ct.array_wrap_from_real_to_model(
            real_working_area)

        # рисуем базовую форму
        draw_polygon(model_ax, model_rezonator,
                     edgecolor='black', facecolor='none')

        # рисуем цели
        self._patches = [draw_model(model_ax, model_view.target(i))
                         for i in range(2)]

        # рисуем запрещенную область
        draw_polygon(model_ax, model_forbidden_area, color='magenta')

        # рисуем рабочую область
        draw_polygon(model_ax, model_working_area,
                     edgecolor='blue', facecolor='none')

        # текущая точка
        model_pos = ct.wrap_from_workzone_relative_to_model(initial_pos)
        self._model_tail = create_tail(model_ax, move_history_size, model_pos)

        # Установка границ по осям X и Y чтобы видно было только рабочую область
        model_ax.set_xlim(min(model_working_area[:, 0]),
                          max(model_working_area[:, 0]))
        model_ax.set_ylim(min(model_working_area[:, 1]),
                          max(model_working_area[:, 1]))

        # ----------------------------------------

        self._info_ax = info_ax
        self._info_ax.set_title('Info')

        self._freq_gistory_curve, = info_ax.plot([], [], 'b-o')

    def __call__(self, input: dict) -> None:
        start_wz: tuple[float, float] = input['move_history'][0][:2]
        start_model = self._ct.wrap_from_workzone_relative_to_model(
            WorkzoneRelativeCoordinates(*start_wz))

        # Траектория
        for wz_curve, model_curve, move_history_item in zip(self._wz_tail, self._model_tail, input['move_history']):
            dest_x, dest_y, S, F = move_history_item
            color = (F, 0, 0, S)
            wz_curve.set_data([start_wz[0], dest_x], [start_wz[1], dest_y])
            wz_curve.set_color(color)  # type: ignore

            model_pos = self._ct.wrap_from_workzone_relative_to_model(
                WorkzoneRelativeCoordinates(dest_x, dest_y))

            model_curve.set_data([start_model[0], model_pos[0]], [
                                 start_model[1], model_pos[1]])
            model_curve.set_color(color)  # type: ignore

            start_wz = move_history_item[:2]
            start_model = model_pos

        # Мишень
        for target in range(2):
            target_colors = self._model_view.target_color_map(target)
            for row in zip(self._patches[target], target_colors):
                for patch, color in zip(*row):
                    patch.set_facecolor(color)

        # Инфо
        freq_history = input['freq_history']

        x = np.linspace(0.0, 1.0, len(freq_history))
        self._freq_gistory_curve.set_data(x, freq_history)
        self._info_ax.relim()
        self._info_ax.autoscale_view()

        plt.pause(0.001)


if __name__ == "__main__":
    import sys

    from constants_v2 import *

    manual = len(sys.argv) > 1

    f, ax = plt.subplots(1, 3)

    rezonator = RezonatorModel(power_threshold=POWER_THRESHOLD)
    initial_pos = WorkzoneRelativeCoordinates(0.0, 1.0)

    # Генерируем случайное смещение и случайный угол поворота
    offset = (np.random.random() * 0.3, np.random.random() * 0.5)
    angle = np.random.random() * 20 - 10
    print('offset: {}, angle: {}'.format(offset, angle))

    rez = Rezonator.load()
    coord_transformer = CoordinateTransformer(rez, (0, 0), offset, angle)

    if manual:
        controller = ManualController()
    else:
        NNController.init_model(F_HISTORY_SIZE, MOVE_HISTORY_SIZE)
        weights = NNController.shuffled_weights()
        controller = NNController(weights)

    sim = Simulator(rezonator_model=rezonator,
                    controller_v2=controller,
                    coord_transformer=coord_transformer,
                    fs_transformer=FSTransformer(255.0, MAX_F),
                    laser_power=LASER_POWER,
                    freqmeter_period=FREQMETER_PERIOD,
                    modeling_period=SIM_CYCLE_TIME,
                    freq_history_size=F_HISTORY_SIZE,
                    initial_wz_pos=initial_pos)

    stop_detector = SimStopDetector(timeout=SIM_TIMEOUT,
                                    history_len_s=SIM_TIMEOUT,
                                    min_path=0.01,
                                    min_avg_speed=MIN_AVG_SPEED,
                                    min_laser_power=POWER_THRESHOLD * 0.5,
                                    max_temperature=MAX_T,
                                    self_grade_epsilon=0.01,
                                    start_energy=START_ENERGY,
                                    energy_consumption_pre_1=ENERGY_CONSUMPTION_PRE_1,
                                    energy_income_per_hz=ENERGY_INCOME_PER_HZ,
                                    energy_fixed_tax=ENERGY_FIXED_TAX,
                                    incum_function=gen_sigmoid(k=5.0, x_offset_to_right=0.2),
                                    start_timestamp=0.0)

    grader = ControllerGrager(dest_freq_ch=DEST_FREQ_CH,
                              f_penalty=gen_sigmoid(
                                  k=LASER_POWER, x_offset_to_right=0.2),  # экспериментальные параметры
                              max_temperature=MAX_T,
                              grade_weights=np.array(FITNES_WEIGHTS))

    model = rezonator.get_model_view(offset, angle)
    input_display = ControllerInputDisplay(
        *ax, rez, model, coord_transformer,  # type: ignore
        move_history_size=MOVE_HISTORY_SIZE, initial_pos=initial_pos
    )

    plt.show(block=False)

    stop_condition = sim.perform_modeling(stop_detector, input_display)

    rm = rezonator.get_metrics()
    total, g = grader.get_grade(rm, stop_detector.summary(), stop_condition)
    print(
        f"Done {stop_condition} ({total}); Fd:{g[0]:.2f}, fzp:{g[1]:.6f} ({rm['penalty_energy']}), db:{g[2]:.2f}, sg:{g[3]:.2f}, Tmax:{g[4]:.2f}, Va:{g[5]:.2f}, t:{g[6]:.2f}, S:{g[7]:.2f}, E:{g[8]:.2f}, scg:{g[9]:.2f}"
    )
    sf, ax = plt.subplots(1, 1)
    stop_detector.plot_summary(ax)
    plt.show(block=True)
