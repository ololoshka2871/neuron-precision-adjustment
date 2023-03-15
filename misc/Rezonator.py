from misc.common import polygon_area


import numpy as np


import json


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