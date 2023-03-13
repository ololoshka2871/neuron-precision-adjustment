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


class Rezonator(dict):
    """
    Размеры частей резонатора в миллиметрах
    - rezonator - полигон тела резонатора
    - targets - полигоны целей
    - working_area - полигоны рабочей области
    - forbidden_area - полигоны запрещенной области
    """

    def __init__(self, rezonator, targets, working_area, forbidden_area):
        self['rezonator'] = rezonator
        self['targets'] = targets
        self['working_area'] = working_area
        self['forbidden_area'] = forbidden_area

    @staticmethod
    def load():
        with open('rezonator.json', 'r') as f:
            data = json.load(f)
            return Rezonator(rezonator=np.array(data['rezonator']),
                             targets=[np.array(target)
                                      for target in data['targets']],
                             working_area=np.array(data['working_area']),
                             forbidden_area=np.array(data['forbidden_area']))

    @property
    def body_square(self) -> float:
        """
        Функция вычисляет площадь тела резонатора в мм^2
        """
        return polygon_area(self['rezonator'])

    def body_volume(self, thikness: float) -> float:
        """
        Функция вычисляет объем тела резонатора с заданной толщиной
        """
        return self.body_square * thikness

    @property
    def target_zone_size(self) -> tuple[float, float]:
        """
        Функция возвращает размеры зоны целей
        """
        tgt = self['targets'][0]
        return tgt[2] - tgt[0]

    @property
    def work_zone_size(self) -> tuple[float, float]:
        """
        Функция возвращает размеры рабочей зоны
        """
        return self['working_area'][2] - self['working_area'][0]

    @property
    def work_zone_base(self) -> tuple[float, float]:
        """
        Функция возвращает базовую точку рабочей зоны
        """
        return self['working_area'][0]

    def get_target_base_point(self, target_index: int) -> tuple[float, float]:
        """
        Функция возвращает базовую точку для заданной цели
        """
        tgt = self['targets'][target_index]
        return tgt[0]

    @property
    def work_zone_center_pos(self) -> tuple[float, float]:
        """
        Функция возвращает центральную точку рабочей зоны
        """
        return self.work_zone_base[0] + self.work_zone_size[0] / 2, \
            self.work_zone_base[1] + self.work_zone_size[1] / 2
              

def gen_sigmoid(A=1.0, k=1.0, x_offset_to_right=0.0):
    def sigmoid(x: float) -> float:
        return A / (1.0 + np.exp(-k * (x - x_offset_to_right)))
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
