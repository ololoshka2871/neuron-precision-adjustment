from misc.Rezonator import Rezonator
from misc.common import create_tail, draw_polygon
from misc.coordinate_transformer import CoordinateTransformer, WorkzoneRelativeCoordinates
from models.adjust_zone_model import draw_model
from models.rezonator_model import ModelView
from parameters_v3 import np


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


class ProgressDisplay:
    def __init__(self, wz_ax: Axes, model_ax: Axes, info_ax: Axes,
                 rez: Rezonator, model_view: ModelView, ct: CoordinateTransformer,
                 move_history_size: int, initial_pos: WorkzoneRelativeCoordinates,
                 possible_freq_adjust: float):
        self._ct = ct
        self._model_view = model_view
        self._possible_freq_adjust = possible_freq_adjust

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
        freq_history = input['freq_history'] * self._possible_freq_adjust

        x = np.linspace(0.0, len(freq_history), len(freq_history))
        self._freq_gistory_curve.set_data(x, freq_history)
        self._info_ax.relim()
        self._info_ax.autoscale_view()

        plt.pause(0.001)