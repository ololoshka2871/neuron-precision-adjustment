from enum import Enum
from typing import Optional
from matplotlib.transforms import Affine2D
import numpy as np
import math

import pygame

from gym_quarz.envs.quartz6 import ActionSpace
from misc.coordinate_transformer import CoordinateTransformer, WorkzoneRelativeCoordinates


class States(Enum):
    # поиск торчащего уголка резонатора, просто змейкой движемся вниз пока частота не изменится.
    # Ждем после заждого прохода чтобы произошло измерение
    FIND_CORNER = 0

    # отступаем от уголка, чтобы не было перекрытия сигналов
    RETREAT = 1

    # определяем касание левой стороны (положительный угол)
    DETECT_LEFT_SIDE = 2

    # определяем касание правой стороны (отрицательный угол)
    DETECT_RIGHT_SIDE = 3

    # поворот на угол на 1 шаг больший чем правая сторона
    MOVE_RIGHT_SIDE_ONE_LEFT = 4

    # постепенно вращаем траекторию влево пока не будет обнаружена реакция
    # Угол предыдущего шага и есть примерно равный угру наклона резонатора
    ROTATE_LEFT_FIND_REACTION = 5

    # Вычисление позиции резонатора по пробным точкам
    CALC_REZONATOR_POS = 6

    # Поворот к найденному углу положения резонатора
    ROTATE_TO_REZONATOR_ANGLE = 7

    # быстрый спуск вниз до зоны настройки
    FAST_FORWARD_TARGET_ZONE = 8

    # Наклон резонатора найден, производим обработку до получения требуемого результата
    WORK_STEPS = 9

    # pause
    PAUSE = 90

    # конец
    DONE = 99


class FindCornerOp(Enum):
    MOVE_HORISONTAL = 0
    MOVE_WAIT = 1
    MOVE_DOWN = 2


class FindSideOp(Enum):
    ROTATE = 0
    MOVE_HORISONTAL = 1
    MOVE_WAIT = 2


class RLFindReactionOp(Enum):
    MOVE_HORISONTAL = 0
    MOVE_WAIT = 1
    ROTATE = 2


class WorkStepsOp(Enum):
    MOVE_DOWN = 0
    MOVE_HORISONTAL = 1
    MOVE_WAIT = 2
    COOLING = 3


class AlgorithmicController:
    """
    Контроллер алгоритмический
    """

    def __init__(self,
                 angle_change_step: float,
                 angle_limit: float,
                 acuracy_hz: float = 0.1,
                 wait_count_betwen_measurements: int = 4,
                 freq_minimal_change: float = 0.1,
                 freq_minimal_change_cooling: Optional[float] = None,
                 retreat_steps: int = 2,
                 fast_forward_steps: Optional[int] = None,
                 work_steps_hor_per1_vert: int = 1):
        """
        :param angle_change_step: шаг изменения угла в градусах мделью
        :param angle_limit: максимальный угол в градусах наклона оси реза
        :param acuracy_hz: требуемая точность настройки в Гц
        :param wait_count_betwen_measurements: количество шагов ожидания чтобы получить новое измерение
        :param freq_minimal_change: минимальное изменение частоты считаемое за изменеие
        :param retreat_steps: количество шагов отступа вверх после обнаружения уголка
        :param work_steps_hor_per1_vert: количество шагов в горизонтальном направлении на 1 шаг в вертикальном при обработке
        """
        self._wait_count_betwen_measurements = wait_count_betwen_measurements
        self._freq_minimal_change = freq_minimal_change
        self._angle_change_step = angle_change_step
        self._freq_minimal_change_cooling = freq_minimal_change_cooling if freq_minimal_change_cooling is not None else freq_minimal_change
        self._angle_limit = angle_limit
        self._retreat_steps = retreat_steps
        self._work_steps_hor_per1_vert = work_steps_hor_per1_vert
        self._acuracy_hz = acuracy_hz
        self._fast_forward_steps = fast_forward_steps if fast_forward_steps is not None else retreat_steps

        AlgorithmicController.reset(self)

    def reset(self):
        self._state = States.FIND_CORNER
        self._find_corner_op = FindCornerOp.MOVE_HORISONTAL
        self._find_corner_wait_count = 0

        self._retreat_step_num = 0

        self._finde_side_op = FindSideOp.ROTATE
        self._find_side_wait_count = 0

        self._rotate_left_find_reaction_op = RLFindReactionOp.MOVE_HORISONTAL
        self._rotate_left_find_reaction_wait_count = 0

        self._fast_forward_step = 0

        self._work_steps_op = WorkStepsOp.MOVE_DOWN
        self._work_steps_wait_count = 0
        self._work_steps_hor_steps = 0

        self._corner_level = None
        self._left_side = None
        self._right_side = None
        self._rezonator_pos = None

    def sample_action(self, prev_observation, observation):
        match self._state:
            case States.FIND_CORNER:
                return self._find_cornet_step(prev_observation, observation)
            case States.RETREAT:
                return self._retreat_step(prev_observation, observation)
            case States.DETECT_LEFT_SIDE:
                return self._detect_side(prev_observation, observation, angle=self._angle_limit)
            case States.DETECT_RIGHT_SIDE:
                return self._detect_side(prev_observation, observation, angle=-self._angle_limit)
            case States.MOVE_RIGHT_SIDE_ONE_LEFT:
                return self._move_right_side_one_left(prev_observation, observation)
            case States.ROTATE_LEFT_FIND_REACTION:
                return self._rotate_left_find_reaction(prev_observation, observation)
            case States.CALC_REZONATOR_POS:
                return self._calc_rezonator_pos(prev_observation, observation)
            case States.ROTATE_TO_REZONATOR_ANGLE:
                return self._rotate_to_rezonator_angle(prev_observation, observation)
            case States.FAST_FORWARD_TARGET_ZONE:
                return self._fast_forward_target_zone(prev_observation, observation)
            case States.WORK_STEPS:
                return self._work_steps(prev_observation, observation)
            case States.PAUSE:
                return ActionSpace.DO_NOTHING.value
            case States.DONE:
                return ActionSpace.END_EPISODE.value
            case _:
                ValueError("Unknown state")

    def _detect_touch(self, prev_observation, observation) -> tuple[bool, bool]:
        if observation[1] == prev_observation[1]:
            return False, False
        else:
            freq_change_change = observation[1] - prev_observation[1]
            # Если частота упала больше чем на _freq_minimal_change, значит мы коснулись резонатора
            return freq_change_change < -self._freq_minimal_change, True

    def _find_cornet_step(self, prev_observation, observation):
        detected, updated = self._detect_touch(prev_observation, observation)
        if not updated:
            return ActionSpace.DO_NOTHING.value

        if detected:
            self._state = States.RETREAT
            self._corner_level = observation[0]
            return ActionSpace.DO_NOTHING.value

        match self._find_corner_op:
            case FindCornerOp.MOVE_HORISONTAL:
                self._find_corner_op = FindCornerOp.MOVE_WAIT
                return ActionSpace.MOVE_HORIZONTAL.value
            case FindCornerOp.MOVE_WAIT:
                self._find_corner_wait_count += 1
                if self._find_corner_wait_count == self._wait_count_betwen_measurements:
                    self._find_corner_wait_count = 0
                    self._find_corner_op = FindCornerOp.MOVE_DOWN
                return ActionSpace.DO_NOTHING.value
            case FindCornerOp.MOVE_DOWN:
                self._find_corner_op = FindCornerOp.MOVE_HORISONTAL
                return ActionSpace.MOVE_DOWN.value

    def _retreat_step(self, prev_observation, observation):
        self._retreat_step_num += 1
        if self._retreat_step_num == self._retreat_steps:
            self._state = States.DETECT_LEFT_SIDE
            self._finde_side_op = FindSideOp.ROTATE
            self._find_side_wait_count = 0

        return ActionSpace.MOVE_UP.value

    def _detect_side(self, prev_observation, observation, angle: float):
        match self._finde_side_op:
            case FindSideOp.ROTATE:
                if angle > 0.0:
                    if observation[4] < angle:
                        return ActionSpace.INCRESE_ANGLE.value
                    else:
                        pass  # drop down
                else:
                    if observation[4] > angle:
                        return ActionSpace.DECREASE_ANGLE.value
                    else:
                        pass  # drop down
                self._finde_side_op = FindSideOp.MOVE_HORISONTAL
                self._find_side_wait_count = 0
                return ActionSpace.DO_NOTHING.value
            case FindSideOp.MOVE_HORISONTAL:
                self._finde_side_op = FindSideOp.MOVE_WAIT
                return ActionSpace.MOVE_HORIZONTAL.value
            case FindSideOp.MOVE_WAIT:
                detected, updated = self._detect_touch(
                    prev_observation, observation)
                if not updated:
                    return ActionSpace.DO_NOTHING.value
                if detected:
                    side_data = {
                        'angle': observation[4], 'offset': observation[0], 'final': False}
                    if angle > 0.0:
                        # left side
                        self._left_side = side_data
                        self._state = States.DETECT_RIGHT_SIDE
                        self._finde_side_op = FindSideOp.ROTATE
                        return ActionSpace.DO_NOTHING.value
                    else:
                        # right side
                        self._right_side = side_data
                        self._state = States.MOVE_RIGHT_SIDE_ONE_LEFT
                        return ActionSpace.DO_NOTHING.value
                else:
                    self._find_side_wait_count += 1
                    if self._find_side_wait_count > self._wait_count_betwen_measurements:
                        # not found
                        zero_side_data = {'angle': 0.0,
                                          'offset': observation[0], 'final': True}
                        if angle > 0.0:
                            self._left_side = zero_side_data
                            self._state = States.DETECT_RIGHT_SIDE
                            self._finde_side_op = FindSideOp.ROTATE
                        else:
                            self._right_side = zero_side_data
                            self._state = States.MOVE_RIGHT_SIDE_ONE_LEFT
                    return ActionSpace.DO_NOTHING.value

    def _move_right_side_one_left(self, prev_observation, observation):
        assert self._right_side is not None

        if observation[4] < self._right_side['angle'] + self._angle_change_step:
            return ActionSpace.INCRESE_ANGLE.value
        else:
            self._state = States.ROTATE_LEFT_FIND_REACTION
            return ActionSpace.DO_NOTHING.value

    def _rotate_left_find_reaction(self, prev_observation, observation):
        assert self._left_side is not None
        assert self._right_side is not None

        match self._rotate_left_find_reaction_op:
            case RLFindReactionOp.MOVE_HORISONTAL:
                if observation[4] < self._left_side['angle']:
                    self._rotate_left_find_reaction_op = RLFindReactionOp.MOVE_WAIT
                    self._rotate_left_find_reaction_wait_count = 0
                    return ActionSpace.MOVE_HORIZONTAL.value
                else:
                    self._left_side = {
                        'angle': observation[4], 'offset': observation[0], 'final': True}
                    self._state = States.CALC_REZONATOR_POS
                    return ActionSpace.DO_NOTHING.value
            case RLFindReactionOp.MOVE_WAIT:
                detected, updated = self._detect_touch(
                    prev_observation, observation)
                if not updated:
                    return ActionSpace.DO_NOTHING.value
                if detected:
                    # found
                    if self._right_side['angle'] != 0.0 and not self._right_side['final']:
                        # реакция была и продолжается - обновить правый угол
                        self._right_side['angle'] = observation[4]  # update angle
                        self._rotate_left_find_reaction_op = RLFindReactionOp.ROTATE  # next step
                    else:
                        # реакции не было и появилась - значит мы нашли левый угол
                        self._left_side = {
                            'angle': observation[4], 'offset': observation[0], 'final': True}
                        self._state = States.CALC_REZONATOR_POS
                else:
                    if not self._right_side['final']:
                        # Реакция была и пропала - значит мы нашли правый угол
                        self._right_side = {
                            'angle': observation[4], 'offset': observation[0], 'final': True}
                        if self._left_side['final']:  # а левый угол уже был найден?
                            self._state = States.CALC_REZONATOR_POS
                        else:
                            self._rotate_left_find_reaction_op = RLFindReactionOp.ROTATE
                    else:
                        # not found and was final - continue
                        self._rotate_left_find_reaction_wait_count += 1
                        if self._rotate_left_find_reaction_wait_count > self._wait_count_betwen_measurements:
                            # not found, next step
                            self._rotate_left_find_reaction_op = RLFindReactionOp.ROTATE
                return ActionSpace.DO_NOTHING.value
            case RLFindReactionOp.ROTATE:
                self._rotate_left_find_reaction_op = RLFindReactionOp.MOVE_HORISONTAL
                return ActionSpace.INCRESE_ANGLE.value

    def _calc_rezonator_pos(self, prev_observation, observation):
        assert self._left_side is not None
        assert self._right_side is not None

        if self._left_side['angle'] != 0 and self._right_side['angle'] != 0:
            self._rezonator_pos = {'angle': (self._left_side['angle'] + self._right_side['angle']) / 2.0,
                                'offset': (self._left_side['offset'] + self._right_side['offset']) / 2.0,
                                'final': True}
        elif self._left_side['angle'] == 0:
            self._rezonator_pos = self._right_side
        else:
            self._rezonator_pos = self._left_side

        # Округлить угол до ближайшего кратного angle_change_step
        self._rezonator_pos['angle'] = round(
            self._rezonator_pos['angle'] / self._angle_change_step) * self._angle_change_step

        self._left_side = None
        self._right_side = None
        self._state = States.ROTATE_TO_REZONATOR_ANGLE
        return ActionSpace.DO_NOTHING.value

    def _rotate_to_rezonator_angle(self, prev_observation, observation):
        assert self._rezonator_pos is not None

        if abs(observation[4] - self._rezonator_pos['angle']) < 0.1:
            self._state = States.FAST_FORWARD_TARGET_ZONE
            return ActionSpace.DO_NOTHING.value
        elif observation[4] < self._rezonator_pos['angle']:
            return ActionSpace.INCRESE_ANGLE.value
        else:
            return ActionSpace.DECREASE_ANGLE.value

    def _fast_forward_target_zone(self, prev_observation, observation):
        self._fast_forward_step += 1
        if self._fast_forward_step == self._fast_forward_steps:
            self._state = States.WORK_STEPS
            self._finde_side_op = FindSideOp.ROTATE
            self._find_side_wait_count = 0
            return ActionSpace.DO_NOTHING.value
        else:
            return ActionSpace.MOVE_DOWN.value

    def _work_steps(self, prev_observation, observation):
        match self._work_steps_op:
            case WorkStepsOp.MOVE_DOWN:
                self._work_steps_op = WorkStepsOp.MOVE_HORISONTAL
                self._work_steps_hor_steps = 0
                return ActionSpace.MOVE_DOWN.value
            case WorkStepsOp.MOVE_HORISONTAL:
                self._work_steps_op = WorkStepsOp.MOVE_WAIT
                self._work_steps_wait_count = 0
                self._work_steps_hor_steps += 1
                return ActionSpace.MOVE_HORIZONTAL.value
            case WorkStepsOp.MOVE_WAIT:
                detected, updated = self._detect_touch(
                    prev_observation, observation)
                if not updated:
                    return ActionSpace.DO_NOTHING.value
                if detected:
                    # hit! Ждем охлаждение
                    self._work_steps_op = WorkStepsOp.COOLING
                else:
                    # не дошли
                    self._work_steps_wait_count += 1
                    if self._work_steps_wait_count > self._wait_count_betwen_measurements:
                        # next step
                        self._work_steps_op = WorkStepsOp.MOVE_DOWN
                return ActionSpace.DO_NOTHING.value
            case WorkStepsOp.COOLING:
                current_freq_change = observation[1]
                freq_change_change = current_freq_change - prev_observation[1]
                target = observation[2]
                if current_freq_change > target:
                    # уже перебор! Стоп!
                    self._state = States.DONE
                elif freq_change_change < self._freq_minimal_change_cooling:
                    # охлаждение закончено
                    if target - current_freq_change < self._acuracy_hz:  # достаточно точно?
                        self._state = States.DONE
                    else:
                        self._work_steps_op = WorkStepsOp.MOVE_DOWN \
                            if self._work_steps_hor_steps == self._work_steps_hor_per1_vert \
                            else WorkStepsOp.MOVE_HORISONTAL
                return ActionSpace.DO_NOTHING.value

    def render_callback(self, canvas, world_transform: Affine2D, ct: CoordinateTransformer):
        def draw_line(linedata, color=None):
            k, b, final = math.atan(math.radians(
                linedata['angle'])), linedata['offset'], linedata['final']
            color = color if color is not None else ((0, 255, 0) if final else (0, 0, 255))
            start_pos = world_transform.transform(
                ct.wrap_from_workzone_relative_to_model(WorkzoneRelativeCoordinates(-1.0, k * -1.0 + b)).tuple())
            end_pos = world_transform.transform(
                ct.wrap_from_workzone_relative_to_model(WorkzoneRelativeCoordinates(1.0, k * 1.0 + b)).tuple())
            pygame.draw.line(canvas, start_pos=start_pos, end_pos=end_pos, width=1,  # type: ignore
                             color=color)

        pygame.Surface.lock(canvas)

        if self._right_side is not None:
            draw_line(self._right_side)

        if self._left_side is not None:
            draw_line(self._left_side)

        if self._rezonator_pos is not None:
            draw_line(self._rezonator_pos, (150, 128, 0))

        pygame.Surface.unlock(canvas)

    @property
    def rezonator_angle(self):
        return self._rezonator_pos['angle'] if self._rezonator_pos is not None else None
