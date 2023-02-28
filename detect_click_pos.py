#!/usr/bin/env python

import time

import numpy as np
import matplotlib.pyplot as plt

from generate_playground import generate_playground
from common import draw_polygon_ext_coords, draw_polyline
from inside_detector import is_point_inside_polygon


if __name__ == '__main__':
    # Генерируем случайное смещение и случайный угол поворота
    offset = (np.random.random() * 0.3, np.random.random() * 0.5)
    angle = np.random.random() * 20 - 10
    print('offset: {}, angle: {}'.format(offset, angle))

    playground = generate_playground(offset=offset, angle=angle)

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
            elif  is_point_inside_polygon(click, playground['targets'][0]):
                print('<1> target 1')
            elif is_point_inside_polygon(click, playground['targets'][1]):
                print('<2> target 2')
            else:
                print('<=> dummy rezonator area')
        else:
            print('< > outside rezonator')

        print('Detection tooks: {}ms'.format((time.time() - start)*1000))
