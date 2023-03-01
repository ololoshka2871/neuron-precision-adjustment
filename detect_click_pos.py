#!/usr/bin/env python

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from generate_playground import generate_playground
from common import draw_polygon
from inside_detector import is_point_inside_polygon

def unmap_from_target(obj_base_pos, obj_size, pos, transform_matrix):
    """
    Функция преобразует координаты точки из системы координат резонатора в систему координат цели в пределах 0.0..1.0
    :param obj_base_pos: координаты базовой точки объекта
    :param obj_size: размер объекта
    :param pos: координаты точки в глобальной системе координат
    :param transform_matrix: матрица преобразования объекта цели
    """

    # обратная матрица преобразования
    inv_transform_matrix = np.linalg.inv(transform_matrix)

    # обратное преобразование точки
    ext_pos = np.array([pos[0], pos[1], 1])
    unwraped_point = np.dot(inv_transform_matrix, ext_pos)

    # вычесть базовую точку объекта из позиции
    unwraped_point_relative = unwraped_point[0:2] - np.array(obj_base_pos)

    # нормализовать координаты
    return unwraped_point_relative / np.array(obj_size)


def adj_weight_gradient(pos):
    return pos[1]


if __name__ == '__main__':
    f, ax = plt.subplots(1, 1)

    # Генерируем случайное смещение и случайный угол поворота
    offset = (np.random.random() * 0.3, np.random.random() * 0.5)
    angle = np.random.random() * 20 - 10
    print('offset: {}, angle: {}'.format(offset, angle))

    playground = generate_playground(offset=offset, angle=angle)

    original_target = playground['original']['targets'][0]
    original_target_size = original_target[2] - original_target[0]

    # рисуем базовую форму
    draw_polygon(ax, playground['rezonator'], edgecolor='black', facecolor='none')

    # рисуем цель
    draw_polygon(ax, playground['targets'][0], color='green')
    draw_polygon(ax, playground['targets'][1], color='green')

    # рисуем запрещенную область
    draw_polygon(ax, playground['forbidden_area'], color='magenta')

    # рисуем рабочую область
    draw_polygon(ax, playground['working_area'], edgecolor='blue', facecolor='none')

    # Установка границ по осям X и Y чтобы видно было только рабочую область
    ax.set_xlim(playground['working_area'][0][0], playground['working_area'][1][0])
    ax.set_ylim(playground['working_area'][1][1], playground['working_area'][2][1])

    plt.draw()

    while True:
        click = f.ginput(show_clicks=True)
        if not click:
            exit(0)

        click = click[0]
        print('point clicked: x:{}, y:{}'.format(click[0], click[1]))

        start = time.time()

        # Проверяем, попал ли клик в область резонатора
        if is_point_inside_polygon(click, playground['rezonator']):
            if is_point_inside_polygon(click, playground['forbidden_area']):
                print('<#> forbidden area')
            elif is_point_inside_polygon(click, playground['targets'][0]):
                target_pos = unmap_from_target(playground['original']['targets'][0][0], original_target_size, click, playground['transform_matrix'])
                adj_w = adj_weight_gradient(target_pos)
                print(f'<1> target 1 (original pos: {target_pos}), adj={adj_w}')
            elif is_point_inside_polygon(click, playground['targets'][1]):
                target_pos = unmap_from_target(playground['original']['targets'][1][0], original_target_size, click, playground['transform_matrix'])
                adj_w = adj_weight_gradient(target_pos)
                print(f'<2> target 2 (original pos: {target_pos}), adj={adj_w}')
            else:
                print('<=> dummy rezonator area')
        else:
            print('< > outside rezonator')

        print('Detection tooks: {}ms'.format((time.time() - start)*1000))
