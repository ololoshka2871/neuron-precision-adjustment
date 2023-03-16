import math


class Movment:
    """
    Рассчитывает траекторию движения по координатам точек и скорости движения
    """

    def get_move_time(self, src: tuple[float, float], dst: tuple[float, float], speed: float) -> float:
        """
        Время движения между двумя точками
        :param src: Начальная точка [мм]
        :param dst: Конечная точка [мм]
        :param speed: Скорость движения [мм/мин]
        :return: Время движения [c]
        """
        assert (speed > 0)
        return math.sqrt((dst[0] - src[0]) ** 2 + (dst[1] - src[1]) ** 2) / speed * 60

    def interpolate_move(self, src: tuple[float, float], dst: tuple[float, float], 
                         speed: float, time_step: float) -> tuple[list[float], list[float], list[float]]:
        """
        Вычислить все промежуточные точки движения из точки src в точку dst со скоростью speed и с шагом по времени time_step
        :param src: Начальная точка [мм]
        :param dst: Конечная точка [мм]
        :param speed: Скорость движения [мм/мин]
        :param time_step: Шаг по времени [c]
        :return: Список промежуточных точек движения включая начальную и конечную точки [мм] + время [c]
        """
        move_time = self.get_move_time(src, dst, speed)
        if move_time == 0.0:
            return ([src[0]], [src[1]], [time_step])
        else:
            move_steps_f = move_time / time_step
            move_steps = math.floor(move_steps_f)
            if move_steps == 0:
                return ([src[0], dst[0]], [src[1], dst[1]], [0, time_step])
            else:
                move_pre_step_x = (dst[0] - src[0]) / move_steps
                move_pre_step_y = (dst[1] - src[1]) / move_steps
                steps_x = [src[0] + move_pre_step_x * i for i in range(move_steps)]
                steps_x.append(dst[0])
                steps_y = [src[1] + move_pre_step_y * i for i in range(move_steps)]
                steps_y.append(dst[1])
                time_steps = [time_step * i for i in range(move_steps + 1)]
                return steps_x, steps_y, time_steps

    def interpolat_move_time_limit(self, src: tuple[float, float], dst: tuple[float, float], 
                                   speed: float, time_step: float, time_limit: float) -> tuple[list[float], list[float], list[float]]:
        """
        Вычислить все промежуточные точки движения из точки src в точку dst со скоростью speed и с шагом по времени time_step, 
        если время перемещения не больше time_limit
        :param src: Начальная точка [мм]
        :param dst: Конечная точка [мм]
        :param speed: Скорость движения [мм/мин]
        :param time_step: Шаг по времени [c]
        :param time_limit: Максимальное время перемещения. Траектория обрезается так, что последняя точка оказывается не больше time_limit
        :return: Список промежуточных точек движения включая начальную и конечную точки [мм]
        """
        move_time = self.get_move_time(src, dst, speed)
        if move_time == 0.0:
            return ([src[0]], [src[1]], [time_step])
        elif move_time <= time_limit:
            return self.interpolate_move(src, dst, speed, time_step)  
        else:
            move_steps_f = min(move_time, time_limit) / time_step
            move_steps = math.ceil(move_steps_f)
            if move_steps == 0:
                return ([src[0]], [src[1]], [0])
            
            move_pre_step_x = (dst[0] - src[0]) / move_steps
            move_pre_step_y = (dst[1] - src[1]) / move_steps
            if move_steps <= 1:
                return ([src[0], src[0] + move_pre_step_x * move_steps], [src[1], src[1] + move_pre_step_y * move_steps], [0, time_step])
            else:
                steps_x = [src[0] + move_pre_step_x * i for i in range(move_steps)]
                steps_y = [src[1] + move_pre_step_y * i for i in range(move_steps)]
                time_steps = [time_step * i for i in range(move_steps + 1)]
                return steps_x, steps_y, time_steps