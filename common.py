import numpy as np
from json import JSONEncoder
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D


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


def build_transform_matrix(base_point=(0, 0), angle=0.0, offset=(0, 0)) -> Affine2D:
    """
    Функция строит матрицу преобразования для поворота и смещения
    :param base_point: базовая точка, вокруг которой происходит поворот
    :param angle: угол поворота в градусах
    :param offset: смещение
    :return: матрица преобразования
    """

    return Affine2D().rotate_deg_around(base_point[0], base_point[1], angle).translate(offset[0], offset[1])


def transform_all(objects: list, matrix: Affine2D):
    """
    Функция преобразует все объекты по заданной матрице
    :param objects: список объектов
    :param matrix: матрица преобразования
    :return: список преобразованных объектов
    """
    return [matrix.transform(obj) for obj in objects]


def draw_polygon(axis: plt.Axes, vertexes, **kwargs):
    """
    Функция рисует объект по заданным вершинам
    :param vertexes: вершины объекта
    :return: object
    """
    return axis.add_patch(Polygon(vertexes, closed=True, **kwargs))