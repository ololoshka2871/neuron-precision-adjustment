import numpy as np
from json import JSONEncoder
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def load_rezonator():
    with open('rezonator.json', 'r') as f:
        data = json.load(f)
        data['rezonator'] = np.array(data['rezonator'])
        data['targets'] = [np.array(target) for target in data['targets']]
        data['working_area'] = np.array(data['working_area'])
        data['forbidden_area'] = np.array(data['forbidden_area'])

    return data


def extend_matrix(matrix):
    """
    Функция дополняет матрицу единицами для умножения на матрицу преобразования
    :param matrix: матрица
    :return: дополненная матрица
    """
    return np.vstack((matrix.T, np.ones(matrix.shape[0])))


def build_transform_matrix(base_point=(0, 0), angle=0.0, offset=(0, 0)):
    """
    Функция строит матрицу преобразования для поворота и смещения
    :param base_point: базовая точка, вокруг которой происходит поворот
    :param angle: угол поворота в градусах
    :param offset: смещение
    :return: матрица преобразования
    """

    # угол в радианах
    angle = np.deg2rad(angle)

    # матрица поворота
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])

    # матрица смещения
    translation_matrix = np.array([[1, 0, base_point[0]],
                                   [0, 1, base_point[1]],
                                   [0, 0, 1]])

    # матрица смещения
    offset_matrix = np.array([[1, 0, offset[0]],
                              [0, 1, offset[1]],
                              [0, 0, 1]])

    # полная матрица преобразования смещение, поворот, обратное смещение
    final_matrix = np.dot(offset_matrix, np.dot(
        np.dot(translation_matrix, rotation_matrix), np.linalg.inv(translation_matrix)))

    return final_matrix


def unextended_matrix(matrix):
    """
    Функция убирает дополнительную размерность
    :param matrix: матрица
    :return: матрица без дополнительной размерности
    """
    return matrix.T


def transform_all(objects, matrix):
    """
    Функция преобразует все объекты по заданной матрице
    :param objects: список объектов
    :param matrix: матрица преобразования
    :return: список преобразованных объектов
    """
    return [np.dot(matrix, obj) for obj in objects]


def draw_polyline(vertexes, format="-", color='black'):
    """
    Функция рисует объект по заданным вершинам
    :param vertexes: вершины объекта
    :param color: цвет
    :return: None
    """
    return plt.plot(vertexes[:, 0], vertexes[:, 1], format, color=color)


def draw_polyline_ext_coords(vertexes, format="-", color='black'):
    """
    Функция рисует объект по заданным вершинам
    :param vertexes: вершины объекта
    :param color: цвет
    :return: None
    """
    vertexes = unextended_matrix(vertexes)
    return draw_polyline(vertexes, format, color)


def draw_polygon(vertexes, **kwargs):
    """
    Функция рисует объект по заданным вершинам
    Внимание! Если среди объектов нет curve, то matplotlib не сможет определить лимиты по осям, следует задать их вручную
    `ax.set_xlim([min_x, max_x])`
    `ax.set_ylim([min_y, max_y])`

    :param vertexes: вершины объекта
    :param color: цвет
    :return: None
    """
    polygon = Polygon(vertexes.T, closed=True, **kwargs)
    plt.gca().add_patch(polygon)


def draw_polygon_ext_coords(vertexes, **kwargs):
    """
    Функция рисует объект по заданным вершинам
    Внимание! Если среди объектов нет curve, то matplotlib не сможет определить лимиты по осям, следует задать их вручную
    `ax.set_xlim([min_x, max_x])`
    `ax.set_ylim([min_y, max_y])`

    :param vertexes: вершины объекта
    :param color: цвет
    :return: None
    """
    vertexes = unextended_matrix(vertexes)
    draw_polygon(vertexes[0:2], **kwargs)