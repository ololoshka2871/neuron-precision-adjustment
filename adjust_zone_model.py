#!/usr/bin/env python

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

from generate_playground import generate_playground


class AdjustZoneModel:
    def __init__(self, divizion: tuple, initial_weight=1.0,
                 weigth_multiplicator=lambda pos: 1.0,
                 minimal_power_threshold=lambda pos: 0.0, energy_consume_rate=lambda pos: 1.0):
        self.initial_weight = initial_weight
        self.field = np.array([[initial_weight * weigth_multiplicator((x/divizion[0], y/divizion[1]))
                              for y in range(divizion[1])] for x in range(divizion[0])])
        self.minimal_power_threshold = minimal_power_threshold
        self.energy_consume_rate = energy_consume_rate

    def to_grid(self, base_pos=(0.0, 0.0), size=(1.0, 1.0)):
        w, h = self.field.shape

        size = (size[0] / w, size[1] / h)

        def color(x, y):
            c = self.field[x][y] / self.initial_weight
            return (c, c, c)

        return [[patches.Rectangle((base_pos[0] + x * size[0], base_pos[1] + y * size[1]), size[0], size[1],
                                   linewidth=1, edgecolor=(0, 0, 0), facecolor=color(x, y))
                for y in range(h)] for x in range(w)]
    
    def find_chank(self, pos):
        """
        Функция возвращает чанк, в котором находится точка
        :param pos: координаты точки в пределах 0.0..1.0
        :return: координаты чанка
        """
        w, h = self.field.shape
        return (int(pos[0] * w), int(pos[1] * h))

    def update(self, pos, time, power=1.0):
        chank_pos = self.find_chank(pos)
        current_value = self.field[chank_pos[0]][chank_pos[1]]

        if current_value > 0 and power > self.minimal_power_threshold(pos):
            current_value = max(0, current_value - self.energy_consume_rate(pos) * time * power)
            if current_value == 0:
                print(f'chank {chank_pos} deepleted')
            else:
                print(f'chank {chank_pos} updated to {current_value}')
            self.field[chank_pos[0]][chank_pos[1]] = current_value


def draw_model(axis: plt.Axes, rects) -> list[patches.Patch]:
    return [axis.add_patch(rect) for row in rects for rect in row]


def update_target(position: tuple, target_polygon_real, target_polygon_original, zone: AdjustZoneModel,
                  patches: list[patches.Patch], untransformed_pos, target_size, **update_args) -> bool:
    if is_point_inside_polygon(position, target_polygon_real):
        original_base_pos = target_polygon_original[0]
        target_pos = (untransformed_pos - original_base_pos) / target_size
        zone.update(target_pos, **update_args)

        for patch in patches:
            patch.remove()

        rects = zone.to_grid(original_base_pos, original_target_size)
        patches[:] = draw_model(untr_pg, rects)
        return True
    return False


if __name__ == '__main__':
    from common import draw_polygon, transform_all
    from inside_detector import is_point_inside_polygon

    f, ax = plt.subplots(1, 2)

    pg = ax[0]

    # Генерируем случайное смещение и случайный угол поворота
    offset = (np.random.random() * 0.3, np.random.random() * 0.5)
    angle = np.random.random() * 20 - 10
    print('offset: {}, angle: {}'.format(offset, angle))

    playground = generate_playground(offset=offset, angle=angle)

    current_pos_global = [0, playground['working_area'][2][1]]

    # рисуем базовую форму
    draw_polygon(pg, playground['rezonator'],
                 edgecolor='black', facecolor='none')

    # рисуем цель
    draw_polygon(pg, playground['targets'][0], color='green')
    draw_polygon(pg, playground['targets'][1], color='green')

    # рисуем запрещенную область
    draw_polygon(pg, playground['forbidden_area'], color='magenta')

    # рисуем рабочую область
    draw_polygon(pg, playground['working_area'],
                 edgecolor='blue', facecolor='none')

    # текущая точка
    current_pos, = pg.plot(current_pos_global[0], current_pos_global[1], 'ro')

    # Установка границ по осям X и Y чтобы видно было только рабочую область
    pg.set_xlim(playground['working_area'][0][0]-0.1,
                playground['working_area'][1][0]+0.1)
    pg.set_ylim(playground['working_area'][1][1]-0.1,
                playground['working_area'][2][1]+0.1)

    # ------------------------------------------------------------

    # точность разбиения цели на чанки
    CHANK_GRID_PRECISION = 10

    original_target = playground['original']['targets'][0]
    original_target_size = original_target[2] - original_target[0]
    target_size_ratio = original_target_size[0] / original_target_size[1]

    inverted_transform_matrix = playground['transform_matrix'].inverted()

    untr_pg = ax[1]

    original_rezonator = playground['original']['rezonator']
    original_forbidden_area = playground['original']['forbidden_area']
    un_transformed_work_zone = transform_all(
        [playground['original']['working_area']], inverted_transform_matrix)[0]

    draw_polygon(untr_pg, original_rezonator,
                 edgecolor='black', facecolor='none')
    draw_polygon(untr_pg, original_forbidden_area, color='magenta')
    draw_polygon(untr_pg, un_transformed_work_zone,
                 edgecolor='blue', facecolor='none')

    def weigth_modificator(pos):
        return pos[1]

    zone1 = AdjustZoneModel((int(CHANK_GRID_PRECISION * target_size_ratio),
                            CHANK_GRID_PRECISION), weigth_multiplicator=weigth_modificator)
    rects = zone1.to_grid(
        playground['original']['targets'][0][0], original_target_size)
    patches1 = draw_model(untr_pg, rects)

    zone2 = AdjustZoneModel((int(CHANK_GRID_PRECISION * target_size_ratio),
                            CHANK_GRID_PRECISION), weigth_multiplicator=weigth_modificator)
    rects = zone2.to_grid(
        playground['original']['targets'][1][0], original_target_size)
    patches2 = draw_model(untr_pg, rects)

    # текущая точка
    transformed_pos = transform_all(
        [current_pos_global], inverted_transform_matrix)[0]
    current_pos_transformed, = untr_pg.plot(
        transformed_pos[0], transformed_pos[1], 'ro')

    untr_pg.set_xlim(
        un_transformed_work_zone[0][0]-0.1, un_transformed_work_zone[2][0]+0.1)
    untr_pg.set_ylim(
        un_transformed_work_zone[1][1]-0.1, un_transformed_work_zone[3][1]+0.1)

    plt.show(block=False)

    time = 0.1

    while True:
        click = f.ginput(show_clicks=True)
        if len(click) == 0:
            exit(0)
        else:
            click = click[0]

        current_pos.set_data(click[0], click[1])

        transformed_pos = transform_all([click], inverted_transform_matrix)[0]
        current_pos_transformed.set_data(
            transformed_pos[0], transformed_pos[1])

        if not update_target(click, playground['targets'][0], playground['original']['targets'][0], zone1, patches1,
                             transformed_pos, original_target_size, time=time):
            if not update_target(click, playground['targets'][1], playground['original']['targets'][1], zone2, patches2,
                                 transformed_pos, original_target_size, time=time):
                plt.draw()
                continue

        plt.draw()
