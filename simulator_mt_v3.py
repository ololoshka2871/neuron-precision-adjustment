#!/usr/bin/env python

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from parameters_v3 import *
from misc.Rezonator import Rezonator
from misc.common import draw_polygon, create_tail
from models.adjust_zone_model import draw_model
from misc.coordinate_transformer import CoordinateTransformer, WorkzoneRelativeCoordinates
from misc.f_s_transformer import FSTransformer
from graders.controller_grader_v3 import ControllerGrager
from models.rezonator_model import RezonatorModel, ModelView
from models.sim_stop_detector_v3 import SimStopDetector
from simulators.simulator_v3 import Simulator
from controllers.controller_v3 import NNController


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
            wz_ax, move_history_size - 1, initial_pos.tuple())
        
        self._wz_target, = wz_ax.plot([], [], 'mo')

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
        self._model_tail = create_tail(model_ax, move_history_size - 1, model_pos)

        self._model_target, = model_ax.plot([], [], 'mo')

        # Установка границ по осям X и Y чтобы видно было только рабочую область
        model_ax.set_xlim(min(model_working_area[:, 0]),
                          max(model_working_area[:, 0]))
        model_ax.set_ylim(min(model_working_area[:, 1]),
                          max(model_working_area[:, 1]))

        # ----------------------------------------

        self._info_ax = info_ax
        self._info_ax.set_title('Info')

        self._freq_gistory_curve, = info_ax.plot([], [], 'b-o')

    def __call__(self, input: dict, dest: WorkzoneRelativeCoordinates) -> None:
        # Вся история перевернута тут, наиболее ноая точка в конце
        start_wz: tuple[float, float] = input['move_history'][0, :2]
        start_model = self._ct.wrap_from_workzone_relative_to_model(
            WorkzoneRelativeCoordinates(*start_wz))
        
        # цели
        self._wz_target.set_data(dest.tuple())
        self._model_target.set_data(self._ct.wrap_from_workzone_relative_to_model(dest).tuple())

        # Траектория
        for wz_curve, model_curve, move_history_item in zip(self._wz_tail, self._model_tail, input['move_history'][1:]):
           side, tp, S, F = move_history_item
           color = (F, 0, 0, S)
           wz_curve.set_data([start_wz[0], side], [start_wz[1], tp])
           wz_curve.set_color(color)  # type: ignore

           model_pos = self._ct.wrap_from_workzone_relative_to_model(
               WorkzoneRelativeCoordinates(side, tp))

           model_curve.set_data([start_model[0], model_pos[0]], [
                                start_model[1], model_pos[1]])
           model_curve.set_color(color)  # type: ignore

           start_wz = (side, tp)
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
    from parameters_v3 import *

    f, ax = plt.subplots(1, 3)

    while True:
        params = gen_sim_parameters()
        rezonator = RezonatorModel(
            power_threshold=POWER_THRESHOLD, layer_thikness=params['ag_thikness'])
        adjustment_freq = rezonator.possible_freq_adjust * \
            params['initial_freq_diff']  # [Hz]
        if adjustment_freq > FREQ_PRECISION:
            break

    print('Offset: {}, angle: {}, i_fd: {}, Ag: {} mm'.format(
        params['offset'], params['angle'], params['initial_freq_diff'], params['ag_thikness']))
    print(
        f"Adjustment freq: +{adjustment_freq:.2f} Hz ({rezonator.possible_freq_adjust:.2f} Hz * {params['initial_freq_diff']:.2f})")

    initial_pos = WorkzoneRelativeCoordinates(-1.0, 1.0)  # Верхний левый угол
    rez = Rezonator.load()
    coord_transformer = CoordinateTransformer(
        rez, (0, 0), params['offset'], params['angle'])

    NNController.init_model(F_HISTORY_SIZE, MOVE_HISTORY_SIZE, mean_layers=NN_MEAN_LAYERS)
    weights = NNController.shuffled_weights()
    controller = NNController(weights, save_history=True)

    sim = Simulator(rezonator_model=rezonator,
                    controller_v3=controller,
                    coord_transformer=coord_transformer,
                    fs_transformer=FSTransformer(255.0, MAX_F),
                    laser_power=LASER_POWER,
                    initial_freq_diff=params['initial_freq_diff'],
                    freqmeter_period=FREQMETER_PERIOD,
                    modeling_period=SIM_CYCLE_TIME,
                    freq_history_size=F_HISTORY_SIZE,
                    initial_wz_pos=initial_pos)

    stop_detector = SimStopDetector(timeout=SIM_TIMEOUT,
                                    history_len_s=SIM_TIMEOUT,
                                    max_temperature=MAX_T,
                                    self_grade_epsilon=0.01,
                                    start_timestamp=0.0)

    grader = ControllerGrager(dest_freq_ch=adjustment_freq,
                              f_penalty=f_penalty,
                              max_temperature=MAX_T)

    model = rezonator.get_model_view(params['offset'], params['angle'])
    input_display = ControllerInputDisplay(
       *ax, rez, model, coord_transformer,  # type: ignore
       move_history_size=MOVE_HISTORY_SIZE, initial_pos=initial_pos
    )

    plt.show(block=False)

    stop_condition = sim.perform_modeling(stop_detector, input_display)

    rm = rezonator.get_metrics()
    total, g = grader.get_grade(rm, stop_detector.summary(), stop_condition)
    precision = (1.0 - (adjustment_freq - rm['static_freq_change']) / adjustment_freq) * 100.0
    print(
        f"""Done {stop_condition} Score = {total}:
- Adjust rgade: {g[0]:.2f} @ {rm['static_freq_change']:.2f} Hz/{adjustment_freq:.2f} Hz: {precision:.2f}%,
- Penalty: {g[1]:.6f} @ {rm['penalty_energy']},
- dissbalance: {g[2] * 100:.2f} %,
- Self grade: {g[3]:.2f},
- Tmax: AMBIENT + {g[4]:.2f} *C,
- Avarage speed: {g[5]:.2f},
- Time spent: {SIM_TIMEOUT * g[6]:.2f} s, ({g[6] * 100:.2f} %),
- Total path: {g[7]:.2f},
- Stop condition grade: {g[8]:.2f}"""
    )

    sf, ax = plt.subplots(1, 2)
    stop_detector.plot_summary(ax[0])
    ax[1].imshow(controller.history().T, interpolation='none', cmap='gray', origin='lower')  # type: ignore
    
    plt.show(block=True)
