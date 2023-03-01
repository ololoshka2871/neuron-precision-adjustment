import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from generate_playground import generate_playground


class AdjustZoneModel:
    def __init__(self, divizion: tuple, initial_weight=1.0,
                 weigth_multiplicator=lambda pos: 1.0,
                 minimal_power_threshold=lambda pos: 1.0, energy_cunsume_rate=lambda pos: 1.0):
        self.initial_weight = initial_weight
        self.field = np.array([[initial_weight * weigth_multiplicator((x, y))
                              for y in range(divizion[1])] for x in range(divizion[0])])
        self.minimal_power_threshold = minimal_power_threshold
        self.energy_cunsume_rate = energy_cunsume_rate

    def to_grid(self, base_pos=(0.0, 0.0), size=(1.0, 1.0)):
        w, h = self.field.shape

        size = (size[0] / w, size[1] / h)

        def color(x, y):
            c = self.field[x][y] / self.initial_weight
            return (c, c, c)

        return [[patches.Rectangle((base_pos[0] + x * size[0], base_pos[1] + y * size[1]), size[0], size[1],
                                   linewidth=1, edgecolor=(0, 0, 0), facecolor=color(x, y))
                for y in range(h)] for x in range(w)]


def draw_model(axis: plt.Axes, rects):
    for row in rects:
        for rect in row:
            axis.add_patch(rect)

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)


if __name__ == '__main__':
    from common import draw_polygon, transform_all

    f, ax = plt.subplots(1, 2)

    pg = ax[0]

    # Генерируем случайное смещение и случайный угол поворота
    offset = (np.random.random() * 0.3, np.random.random() * 0.5)
    angle = np.random.random() * 20 - 10
    print('offset: {}, angle: {}'.format(offset, angle))

    playground = generate_playground(offset=offset, angle=angle)

    # рисуем базовую форму
    draw_polygon(pg, playground['rezonator'], edgecolor='black', facecolor='none')

    # рисуем цель
    draw_polygon(pg, playground['targets'][0], color='green')
    draw_polygon(pg, playground['targets'][1], color='green')

    # рисуем запрещенную область
    draw_polygon(pg, playground['forbidden_area'], color='magenta')

    # рисуем рабочую область
    draw_polygon(pg, playground['working_area'], edgecolor='blue', facecolor='none')

    # Установка границ по осям X и Y чтобы видно было только рабочую область
    pg.set_xlim(playground['working_area'][0][0]-0.1, playground['working_area'][1][0]+0.1)
    pg.set_ylim(playground['working_area'][1][1]-0.1, playground['working_area'][2][1]+0.1)

    #------------------------------------------------------------

    # точность разбиения цели на чанки
    CHANK_GRID_PRECISION = 10

    original_target = playground['original']['targets'][0]
    original_target_size = original_target[2] - original_target[0]
    target_size_ratio = original_target_size[0] / original_target_size[1]

    untr_pg = ax[1]

    original_rezonator = playground['original']['rezonator']
    original_forbidden_area = playground['original']['forbidden_area']
    un_transformed_work_zone = transform_all([playground['original']['working_area']], playground['transform_matrix'].inverted())[0]

    draw_polygon(untr_pg, original_rezonator, edgecolor='black', facecolor='none')
    draw_polygon(untr_pg, original_forbidden_area, color='magenta')
    draw_polygon(untr_pg, un_transformed_work_zone, edgecolor='blue', facecolor='none')

    zone = AdjustZoneModel((int(CHANK_GRID_PRECISION * target_size_ratio), CHANK_GRID_PRECISION))
    rects = zone.to_grid(playground['original']['targets'][0][0], original_target_size)
    draw_model(untr_pg, rects)

    zone = AdjustZoneModel((int(CHANK_GRID_PRECISION * target_size_ratio), CHANK_GRID_PRECISION))
    rects = zone.to_grid(playground['original']['targets'][1][0], original_target_size)
    draw_model(untr_pg, rects)

    untr_pg.set_xlim(un_transformed_work_zone[0][0]-0.1, un_transformed_work_zone[2][0]+0.1)
    untr_pg.set_ylim(un_transformed_work_zone[1][1]-0.1, un_transformed_work_zone[3][1]+0.1)

    plt.show()
