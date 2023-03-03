#!/usr/bin/env python

from common import build_transform_matrix, transform_all, Rezonator

rezonator_cached = None

def generate_playground(offset=(0, 0), angle=0.0):
    """
    Функция генерирует тестовую площадку для резонатора
    :param offset: смещение X: 0..0.3, Y: 0..0.5
    :param angle: угол поворота в градусах: -10..+10
    """
    global rezonator_cached
    if not rezonator_cached:
        rezonator_cached = Rezonator.load()

    transformation_matrix = build_transform_matrix(angle=angle, offset=offset)

    elements = [
        rezonator_cached['rezonator'],
        rezonator_cached['targets'][0],
        rezonator_cached['targets'][1],
        rezonator_cached['forbidden_area'],
    ]

    # transform rezonator elemeeents
    elements = transform_all(elements, transformation_matrix)

    return {
        'rezonator': elements[0],
        'targets': elements[1:3],
        'forbidden_area': elements[3],
        'working_area': rezonator_cached['working_area'],

        'original': rezonator_cached,
        'transform_matrix': transformation_matrix,
    }

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from common import draw_polygon

    f, ax = plt.subplots(1, 1)

    while True:
        # Генерируем случайное смещение и случайный угол поворота
        offset = (np.random.random() * 0.3, np.random.random() * 0.5)
        angle = np.random.random() * 20 - 10
        print('offset: {}, angle: {}'.format(offset, angle))

        playground = generate_playground(offset=offset, angle=angle)

        # рисуем базовую форму
        draw_polygon(ax, playground['rezonator'], color='black')
        draw_polygon(ax, playground['targets'][0], color='green')
        draw_polygon(ax, playground['targets'][1], color='green')

        # рисуем запрещенную область
        draw_polygon(ax, playground['forbidden_area'], color='magenta')

        # рисуем рабочую область
        draw_polygon(ax, playground['working_area'], facecolor='none', edgecolor='blue')
        
        # Установка одинакового масштаба по осям X и Y
        ax.axis('equal')

        plt.draw()

        click = plt.ginput(timeout=5)
        if not click:
            exit(0)

        click = click[0]
        print('point clicked: x:{}, y:{}'.format(click[0], click[1]))

        # Очищаем график
        ax.clear()
