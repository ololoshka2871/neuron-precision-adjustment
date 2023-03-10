
import numpy as np

from misc.common import Rezonator, build_transform_matrix


class Coordinates:
    """
    Базовый класс для координат
    """

    def __init__(self, x: float, y: float) -> None:
        self.point = np.array([x, y])

    def __eq__(self, other) -> bool:
        if isinstance(other, Coordinates):
            return np.array_equal(self.point, other.point)
        else:
            return False
    

class RealCoordinates(Coordinates):
    """
    Координаты в локальном пространстве одного резонатора, включают весь резонатор и рабочую зону
    Начальная точка - место крепления резонатора (внизу по центру)
    """  

    @staticmethod
    def from_ndarray(arr: np.ndarray):
        return RealCoordinates(arr[0], arr[1])
    
    def __str__(self):
        return f"RealCoordinates({self.point[0]}, {self.point[1]})"

class ModelCoordinates(Coordinates):
    """
    Модельные координаты в локальном пространстве одного резонатора, включают весь резонатор и рабочую зону
    Распологается так, что резонатор установлен ровно вдоль оси Y, а рабочая зона трансформирована.
    """

    @staticmethod
    def from_ndarray(arr: np.ndarray):
        return ModelCoordinates(arr[0], arr[1])
    
    def __str__(self):
        return f"ModelCoordinates({self.point[0]}, {self.point[1]})"
  

class WorkzoneRelativeCoordinates(Coordinates):
    """
    Относительные координаты в локальном пространстве рабочей зоны одного резонатора
    Начальная точка - центр рабочей зоны
    """

    @staticmethod
    def from_ndarray(arr: np.ndarray):
        return WorkzoneRelativeCoordinates(arr[0], arr[1])
    
    def __str__(self):
        return f"WorkzoneRelativeCoordinates({self.point[0]}, {self.point[1]})"


class CoordinateTransformer:
    """
    Преобразование координат из одной системы в другую
    """

    def __init__(self, resonator: Rezonator, offset = (0, 0), angle = 0.0):
        self._transform = build_transform_matrix(offset, angle)
        self._revers_transform = self._transform.inverted()

    def wrap_from_real_to_model(self, real_coordinates: RealCoordinates) -> ModelCoordinates:
        return ModelCoordinates.from_ndarray(self._revers_transform.transform(real_coordinates.point))

    def wrap_from_model_to_real(self, model_coordinates: ModelCoordinates) -> RealCoordinates:
        pass

    def wrap_from_real_to_workzone_relative(self, real_coordinates: RealCoordinates) -> WorkzoneRelativeCoordinates:
        pass

    def wrap_from_workzone_relative_to_real(self, workzone_relative_coordinates: WorkzoneRelativeCoordinates) -> RealCoordinates:
        pass

    def wrap_from_model_to_workzone_relative(self, model_coordinates: ModelCoordinates) -> WorkzoneRelativeCoordinates:
        pass

    def wrap_from_workzone_relative_to_model(self, workzone_relative_coordinates: WorkzoneRelativeCoordinates) -> ModelCoordinates:
        pass
    