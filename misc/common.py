import numpy as np
from json import JSONEncoder
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def gen_sigmoid(A=1.0, k=1.0, x_offset_to_right=0.0, vertical_shift=0.0):
    def sigmoid(x: float) -> float:
        return A / (1.0 + np.exp(-k * (x - x_offset_to_right))) + vertical_shift
    return sigmoid


def polygon_area(vertices) -> float:
    """
    Для вычисления площади многоугольника можно воспользоваться формулой Гаусса
    """
    x, y = vertices.T
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


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


def limit(v: float, _min: float, _max: float) -> float:
    return max(_min, min(v, _max))


def create_tail(ax, tail_len: int, init_pos):
    return [ax.plot(*init_pos, 'o-')[0] for _ in range(tail_len)]


def normal_dist(x, mean=0.0, sd=1.0):
    """
    Функция вычисляет плотность вероятности нормального распределения
    :param x: точка, в которой вычисляется плотность
    :param mean: математическое ожидание
    :param sd: среднеквадратичное отклонение
    """
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def my_normal(min_val=-0.5, max_val=0.5) -> float:
    return min(max(np.random.normal(), min_val), max_val)