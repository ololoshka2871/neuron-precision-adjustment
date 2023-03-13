#!/usr/bin/env python


import numpy as np

from matplotlib.transforms import Affine2D

from misc.common import Rezonator, build_transform_matrix


class Coordinates:
    """
    Базовый класс для координат
    """

    def __init__(self, x: float, y: float) -> None:
        self.point = np.array([x, y])

    def __eq__(self, other) -> bool:
        if isinstance(other, Coordinates):
            return np.all(np.isclose(self.point, other.point, rtol=1e-10, equal_nan=False)) == True
        else:
            return False

    def offset(self, x: float, y: float):
        return self.point + np.array([x, y])
    
    def tuple(self):
        return tuple(self.point)
    
    def __getitem__(self, key):
        return self.point[key]


class RealCoordinates(Coordinates):
    """
    Координаты в локальном пространстве одного резонатора, включают весь резонатор и рабочую зону
    Начальная точка - место крепления резонатора (внизу по центру)
    """

    @staticmethod
    def from_ndarray(arr: np.ndarray):
        return RealCoordinates(arr[0], arr[1])

    def __repr__(self):
        return f"RealCoordinates({self.point[0]}, {self.point[1]})"


class ModelCoordinates(Coordinates):
    """
    Модельные координаты в локальном пространстве одного резонатора, включают весь резонатор и рабочую зону
    Распологается так, что резонатор установлен ровно вдоль оси Y, а рабочая зона трансформирована.
    """

    @staticmethod
    def from_ndarray(arr: np.ndarray):
        return ModelCoordinates(arr[0], arr[1])

    def __repr__(self):
        return f"ModelCoordinates({self.point[0]}, {self.point[1]})"


class WorkzoneRelativeCoordinates(Coordinates):
    """
    Относительные координаты в локальном пространстве рабочей зоны одного резонатора
    Начальная точка - центр рабочей зоны
    """

    @staticmethod
    def from_ndarray(arr: np.ndarray):
        return WorkzoneRelativeCoordinates(arr[0], arr[1])

    def __repr__(self):
        return f"WorkzoneRelativeCoordinates({self.point[0]}, {self.point[1]})"


class CoordinateTransformer:
    """
    Преобразование координат из одной системы в другую

    real_coordinates - координаты в реальном пространстве станка, базоая точка - точка крепления резонатора, резонатор отстоит от этой точки на offset и повернут на angle
    workzone_relative_coordinates - координаты в локальном пространстве рабочей зоны резонатора, базовая точка - центр рабочей зоны, эти координаты используются для ввода-вывода в нейронную сеть
    model_coordinates - модельные координаты в локальном пространстве резонатора, базовая точка - точка крепления резонатора, но резонатор в этой системе координат стоит ровно вдоль оси Y, 
        а рабочая зона трансформирована. Эти координаты используются для расчета физики нагрева и обработки мишеней

    FAQ:
    Q: Где находится центр рабочей зоны в реальных координатах?
    A: по координатам workzone_center мы вручную подводим лазер к центру закрепленного резонатора и берем получившиеся координаты и так для каждого резонатора.

    Q: Где находится базовая точка модельной системы координат в реальных координатах?
    A: Известно положение work_zone_center_pos в модельных координатах, а также положение центра рабочей зоны в реальных координатах, поэтому можно вычислить базовую точку модельной системы координат.
    """

    def __init__(self, resonator: Rezonator, workzone_center=(0, 0), offset=(0, 0), angle=0.0):
        self._workzone_center = workzone_center

        workzone_center_offset = resonator.work_zone_center_pos

        self._model_base_point = workzone_center[0] - workzone_center_offset[0], \
            workzone_center[1] - workzone_center_offset[1]

        self._real2model_transform = Affine2D() \
            .rotate_deg_around(0, 0, angle) \
            .translate(offset[0] + self._model_base_point[0], offset[1] + self._model_base_point[1]) \
            
        self._real2workzone_relative_transform = Affine2D() \
            .translate(-workzone_center[0], -workzone_center[1]) \
            .scale(2.0 / resonator.work_zone_size[0], 2.0 / resonator.work_zone_size[1])

        self._model2workzone_relative_transform = Affine2D() \
            .rotate_deg_around(0, 0, angle) \
            .translate(-workzone_center[0] + offset[0] + self._model_base_point[0], -workzone_center[1] + offset[1] + self._model_base_point[1]) \
            .scale(2.0 / resonator.work_zone_size[0], 2.0 / resonator.work_zone_size[1])

    def wrap_from_real_to_model(self, real_coordinates: RealCoordinates) -> ModelCoordinates:
        return ModelCoordinates.from_ndarray(self._real2model_transform.inverted().transform(real_coordinates.point))

    def wrap_from_model_to_real(self, model_coordinates: ModelCoordinates) -> RealCoordinates:
        return RealCoordinates.from_ndarray(self._real2model_transform.transform(model_coordinates.point))

    def wrap_from_real_to_workzone_relative(self, real_coordinates: RealCoordinates) -> WorkzoneRelativeCoordinates:
        return WorkzoneRelativeCoordinates.from_ndarray(self._real2workzone_relative_transform.transform(real_coordinates.point))

    def wrap_from_workzone_relative_to_real(self, workzone_relative_coordinates: WorkzoneRelativeCoordinates) -> RealCoordinates:
        return RealCoordinates.from_ndarray(self._real2workzone_relative_transform.inverted().transform(workzone_relative_coordinates.point))

    def wrap_from_model_to_workzone_relative(self, model_coordinates: ModelCoordinates) -> WorkzoneRelativeCoordinates:
        return WorkzoneRelativeCoordinates.from_ndarray(self._model2workzone_relative_transform.transform(model_coordinates.point))

    def wrap_from_workzone_relative_to_model(self, workzone_relative_coordinates: WorkzoneRelativeCoordinates) -> ModelCoordinates:
        return ModelCoordinates.from_ndarray(self._model2workzone_relative_transform.inverted().transform(workzone_relative_coordinates.point))

    def array_wrap_from_model_to_real(self, model_coordinates: np.ndarray) -> np.ndarray:
        return self._real2model_transform.transform(model_coordinates)
    
    @property
    def workzone_center(self) -> tuple[float, float]:
        return self._workzone_center
    
    @property
    def model_base_point(self) -> tuple[float, float]:
        return self._model_base_point
    
    def get_real_working_zone(self, points: np.ndarray) -> np.ndarray:
        return points + self._model_base_point
    
    def array_wrap_from_real_to_model(self, real_coordinates: np.ndarray) -> np.ndarray:
        return self._real2model_transform.inverted().transform(real_coordinates)
    
    def array_wrap_from_real_to_workzone(self, real_coordinates: np.ndarray) -> np.ndarray:
        return self._real2workzone_relative_transform.transform(real_coordinates)
    
    def array_wrap_from_workzone_to_real(self, workzone_relative_coordinates: np.ndarray) -> np.ndarray:
        return self._real2workzone_relative_transform.inverted().transform(workzone_relative_coordinates)
    
    def array_wrap_from_model_to_workzone(self, model_coordinates: np.ndarray) -> np.ndarray:
        return self._model2workzone_relative_transform.transform(model_coordinates)
    
    def array_wrap_from_workzone_to_model(self, workzone_relative_coordinates: np.ndarray) -> np.ndarray:
        return self._model2workzone_relative_transform.inverted().transform(workzone_relative_coordinates)