import numpy as np
import matplotlib.pyplot as plt

from common import build_transform_matrix, transform_all, load_rezonator, draw_polygon


def draw_test():
    f, ax = plt.subplots(1, 1)

    rez = load_rezonator()
    
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
    
    # рисуем базовую форму
    draw_polygon(ax, rezonator, facecolor='none', edgecolor='black')
    draw_polygon(ax, target1, color='black')
    draw_polygon(ax, target2, color='black')

    # рисуем рабочую область
    draw_polygon(ax, working_area, facecolor='none', edgecolor='blue')

    # рисуем запрещенную область
    draw_polygon(ax, forbidden_area, color='magenta')

    #---------------------------

    matrix = build_transform_matrix(angle=0, offset=(0.2, 0.5))

    transformed = transform_all([rezonator, target1, target2, forbidden_area], matrix)

    draw_polygon(ax, transformed[0], edgecolor='red', facecolor='none')
    draw_polygon(ax, transformed[1], color='red')
    draw_polygon(ax, transformed[2], color='red')
    draw_polygon(ax, transformed[3], color='darkred')

    #---------------------------

    matrix = build_transform_matrix(angle=-10, offset=(0.2, 0.5))

    transformed = transform_all([rezonator, target1, target2, forbidden_area], matrix)

    draw_polygon(ax, transformed[0], edgecolor='green', facecolor='none')
    draw_polygon(ax, transformed[1], color='green')
    draw_polygon(ax, transformed[2], color='green')
    draw_polygon(ax, transformed[3], color='darkgreen')

    #---------------------------

    # Установка одинакового масштаба по осям X и Y
    plt.axis('equal')

    plt.show()


if __name__ == '__main__':
    draw_test()