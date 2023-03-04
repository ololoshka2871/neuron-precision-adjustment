#!/usr/bin/env python

from rezonator_model import Rezonator

class Rect:
    def __init__(self, x0: float, y0: float, x1: float, y1: float):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    @staticmethod
    def base_size(pos: tuple[float, float], size: tuple[float, float]):
        return Rect(pos[0], pos[1], pos[0] + size[0], pos[1] + size[1])
    
    @staticmethod
    def from_rezonator(rezonator: Rezonator, center: tuple[float, float]):
        return Rect.base_size(center, rezonator.work_zone_size)

    def __str__(self):
        return f"Rect({self.x0}, {self.y0}, {self.x1}, {self.y1})"

    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def base(self) -> tuple[float, float]:
        return self.x0, self.y0

    def size(self) -> tuple[float, float]:
        return self.x1 - self.x0, self.y1 - self.y0


class WorkZone:
    """
    Рабочая зона одного резонатора
    """

    def __init__(self, global_rect: Rect, max_s: float, max_f: float):
        self._global_rect = global_rect
        self._max_s = max_s
        self._max_f = max_f

        self._center = self._global_rect.center()
        self._size = self._global_rect.size()

    def map_to_global(self, pos: tuple[float, float]) -> tuple[float, float]:
        """
        Преобразование координат из локальной системы координат в глобальную
        :param pos: Координаты в локальной системе координат [-1..1]
        :return: Координаты в глобальной системе координат [мм]
        """
        return self._center[0] + pos[0] * self._size[0] / 2.0, \
               self._center[1] + pos[1] * self._size[1] / 2.0

    def map_from_global(self, pos: tuple[float, float]) -> tuple[float, float]:
        """
        Преобразование координат из глобальной системы координат в локальную
        :param pos: Координаты в глобальной системе координат [мм]
        :return: Координаты в локальной системе координат [-1..1]
        """
        return (pos[0] - self._center[0]) * 2.0 / self._size[0], \
               (pos[1] - self._center[1]) * 2.0 / self._size[1]

    def map_s_to_global(self, s: float) -> float:
        """
        Преобразование мощности лазера из локальной системы координат в глобальную
        :param s: Мощность лазера в локальной системе координат [0..1]
        :return: Мощность лазера в глобальной системе координат [0..max_s]
        """
        return s * self._max_s
    
    def map_s_from_global(self, s: float) -> float:
        """
        Преобразование мощности лазера из глобальной системы координат в локальную
        :param s: Мощность лазера в глобальной системе координат [0..max_s]
        :return: Мощность лазера в локальной системе координат [0..1]
        """
        return s / self._max_s

    def map_f_to_global(self, f: float) -> float:
        """
        Преобразование скорости перемещения из локальной системы координат в глобальную
        :param f: Скорость перемещения в локальной системе координат [0..1]
        :return: Скорость перемещения в глобальной системе координат [0..max_f]
        """
        return f * self._max_f
    
    def map_f_from_global(self, f: float) -> float:
        """
        Преобразование скорости перемещения из глобальной системы координат в локальную
        :param f: Скорость перемещения в глобальной системе координат [0..max_f]
        :return: Скорость перемещения в локальной системе координат [0..1]
        """
        return f / self._max_f
    
    def map_relative_to_local(self, relative_pos: tuple[float, float]) -> tuple[float, float]:
        """
        Преобразование координат из локальной системы координат в локальную
        :param relative_pos: Координаты в локальной системе координат [-1..1]
        :return: Координаты в локальной системе координат (относительно центра) [мм]
        """
        return relative_pos[0] * self._size[0] / 2.0, relative_pos[1] * self._size[1] / 2.0
