#!/usr/bin/env python

import time

import numpy as np
import matplotlib.pyplot as plt

from generate_playground import generate_playground
from common import draw_polygon_ext_coords, draw_polyline, extend_matrix
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


if __name__ == '__main__':
    # Генерируем случайное смещение и случайный угол поворота
    offset = (np.random.random() * 0.3, np.random.random() * 0.5)
    angle = np.random.random() * 20 - 10
    print('offset: {}, angle: {}'.format(offset, angle))

    playground = generate_playground(offset=offset, angle=angle)

    original_target = playground['original']['targets'][0]
    original_target_size = original_target[2] - original_target[0]

    # рисуем базовую форму
    draw_polygon_ext_coords(playground['rezonator'], color='black')
    draw_polygon_ext_coords(playground['targets'][0], color='green')
    draw_polygon_ext_coords(playground['targets'][1], color='green')

    # рисуем запрещенную область
    draw_polygon_ext_coords(playground['forbidden_area'], color='magenta')

    # рисуем рабочую область
    draw_polyline(playground['working_area'], format='-.', color='blue')
    
    # Установка одинакового масштаба по осям X и Y
    plt.axis('equal')

    plt.draw()

    while True:
        click = plt.ginput(show_clicks=True)
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
                print('<1> target 1 (original pos: {})'.format(target_pos))
            elif is_point_inside_polygon(click, playground['targets'][1]):
                target_pos = unmap_from_target(playground['original']['targets'][1][0], original_target_size, click, playground['transform_matrix'])
                print('<2> target 2 (original pos: {})'.format(target_pos))
            else:
                print('<=> dummy rezonator area')
        else:
            print('< > outside rezonator')

        print('Detection tooks: {}ms'.format((time.time() - start)*1000))
