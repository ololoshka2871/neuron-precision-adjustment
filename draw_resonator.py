import numpy as np
import matplotlib.pyplot as plt

from common import extend_matrix, build_transform_matrix, unextended_matrix, transform_all, load_rezonator, draw_object_ext_coords


def draw_test():
    rez = load_rezonator()
    
    # базовая точка - середина в месте крепления (0, 0)
    rezonator = extend_matrix(rez['rezonator'])
    
    # первая ветка
    target1 = extend_matrix(rez['targets'][0])

    # вторая ветка
    target2 = extend_matrix(rez['targets'][1])

    # рабочая область
    working_area = extend_matrix(rez['working_area'])
    
    # Запрещенная область
    forbidden_area = extend_matrix(rez['forbidden_area'])
    
    # рисуем базовую форму
    draw_object_ext_coords(rezonator, color='black', format='--')
    draw_object_ext_coords(target1, color='black')
    draw_object_ext_coords(target2, color='black')

    # рисуем рабочую область
    draw_object_ext_coords(working_area, color='blue', format='-.')

    # рисуем запрещенную область
    draw_object_ext_coords(forbidden_area, color='magenta', format='-')

    #---------------------------

    matrix = build_transform_matrix(angle=0, offset=(0.2, 0.5))

    transformed = transform_all([rezonator, target1, target2, forbidden_area], matrix)

    draw_object_ext_coords(transformed[0], color='red', format='--')
    draw_object_ext_coords(transformed[1], color='red')
    draw_object_ext_coords(transformed[2], color='red')
    draw_object_ext_coords(transformed[3], color='red')

    #---------------------------

    matrix = build_transform_matrix(angle=-10, offset=(0.2, 0.5))

    transformed = transform_all([rezonator, target1, target2, forbidden_area], matrix)

    draw_object_ext_coords(transformed[0], color='green', format='--')
    draw_object_ext_coords(transformed[1], color='green')
    draw_object_ext_coords(transformed[2], color='green')
    draw_object_ext_coords(transformed[3], color='green')

    #---------------------------

    # Установка одинакового масштаба по осям X и Y
    plt.axis('equal')

    plt.show()


if __name__ == '__main__':
    draw_test()