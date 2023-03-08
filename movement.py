import math


class Movment:
    """
    Рассчитывает траекторию движения по координатам точек и скорости движения
    """

    def __init__(self):
        pass

    def get_move_time(self, src: tuple[float, float], dst: tuple[float, float], speed: float) -> float:
        """
        Время движения между двумя точками
        :param src: Начальная точка [мм]
        :param dst: Конечная точка [мм]
        :param speed: Скорость движения [мм/мин]
        :return: Время движения [c]
        """
        return math.sqrt((dst[0] - src[0]) ** 2 + (dst[1] - src[1]) ** 2) / speed * 60

    def interpolate_move(self, src: tuple[float, float], dst: tuple[float, float], speed: float, time_step: float) -> tuple[list[float], list[float]]:
        """
        Вычислить все промежуточные точки движения из точки src в точку dst со скоростью speed и с шагом по времени time_step
        :param src: Начальная точка [мм]
        :param dst: Конечная точка [мм]
        :param speed: Скорость движения [мм/мин]
        :param time_step: Шаг по времени [c]
        :return: Список промежуточных точек движения включая начальную и конечную точки [мм]
        """
        move_time = self.get_move_time(src, dst, speed)
        move_steps_f = move_time / time_step
        move_steps = math.floor(move_steps_f)
        move_pre_step_x = (dst[0] - src[0]) / move_steps
        move_pre_step_y = (dst[1] - src[1]) / move_steps
        steps_x = [src[0] + move_pre_step_x * i for i in range(move_steps)]
        steps_x.append(dst[0])
        steps_y = [src[1] + move_pre_step_y * i for i in range(move_steps)]
        steps_y.append(dst[1])
        return steps_x, steps_y
