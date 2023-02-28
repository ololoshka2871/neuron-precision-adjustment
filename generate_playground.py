
from common import load_rezonator, build_transform_matrix, transform_all, extend_matrix, unextended_matrix

rezonator_cached = None

def generate_playground(offset=(0, 0), angle=0.0):
    """
    Функция генерирует тестовую площадку для резонатора
    :param offset: смещение X: 0..0.3, Y: 0..0.5
    :param angle: угол поворота в градусах: -10..+10
    """
    global rezonator_cached
    if not rezonator_cached:
        rezonator_cached = load_rezonator()

    transformation_matrix = build_transform_matrix(angle=angle, offset=offset)

    elements = [
        extend_matrix(rezonator_cached['rezonator']),
        extend_matrix(rezonator_cached['targets'][0]),
        extend_matrix(rezonator_cached['targets'][1]),
        extend_matrix(rezonator_cached['forbidden_area']),
    ]

    # transform rezonator elemeeents
    elements = transform_all(elements, transformation_matrix)

    return {
        'rezonator': unextended_matrix(elements[0]),
        'targets': list(map(unextended_matrix, elements[1:3])),
        'forbidden_area': unextended_matrix(elements[3]),
        'working_area': rezonator_cached['working_area']
    }

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from common import draw_object

    while True:
        # Генерируем случайное смещение и случайный угол поворота
        offset = (np.random.random() * 0.3, np.random.random() * 0.5)
        angle = np.random.random() * 20 - 10

        playground = generate_playground(offset=offset, angle=angle)
        
        # рисуем базовую форму
        draw_object(playground['rezonator'], color='black')
        draw_object(playground['targets'][0], color='green')
        draw_object(playground['targets'][1], color='green')

        # рисуем запрещенную область
        draw_object(playground['forbidden_area'], color='magenta', format='--')

        # рисуем рабочую область
        draw_object(playground['working_area'], color='blue', format='-.')
        
        # Установка одинакового масштаба по осям X и Y
        plt.axis('equal')

        plt.show()

        click = plt.ginput()

        # clear plot
        plt.clf()
