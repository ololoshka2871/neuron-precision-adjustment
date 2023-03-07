import numpy as np

def is_point_inside_polygon(point: tuple[float, float], polygon):
    """
    Функция проверяет, находится ли точка внутри многоугольника
    :param point: точка
    :param polygon: многоугольник
    """

    # Разбиваем многоугольник на отрезки между вершинами
    segments = []
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        segments.append((p1, p2))

    # Определяем начальную точку для луча, проходящего через заданную точку
    start_point = (np.min(polygon[:, 0]) - 1, point[1])

    # Считаем количество пересечений луча с отрезками многоугольника
    intersection_count = 0
    for segment in segments:
        if intersect(start_point, point, segment[0], segment[1]):
            intersection_count += 1

    # Если количество пересечений нечетное, то точка находится внутри многоугольника
    return intersection_count % 2 == 1


def intersect(a, b, c, d):
    """
    Функция проверяет, пересекаются ли отрезки a-b и c-d
    :param a: начало первого отрезка
    :param b: конец первого отрезка
    :param c: начало второго отрезка
    :param d: конец второго отрезка
    """

    s1_x, s1_y = b[0] - a[0], b[1] - a[1]
    s2_x, s2_y = d[0] - c[0], d[1] - c[1]

    det = (-s2_x * s1_y + s1_x * s2_y)
    if det == 0:
        return False

    s = (-s1_y * (a[0] - c[0]) + s1_x * (a[1] - c[1])) / det
    t = ( s2_x * (a[1] - c[1]) - s2_y * (a[0] - c[0])) / det

    return (s >= 0 and s <= 1 and t >= 0 and t <= 1)