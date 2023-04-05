from enum import Enum
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


class AlgorithmicController:
    """
    Контроллер алгоритмический
    """

    def __init__(self,
                 angle_change_step: float,
                 angle_limit: float,
                 wait_count_betwen_measurements: int = 4,
                 freq_minimal_change: float = 0.1,
                 retreat_steps: int = 1,):
        self._wait_count_betwen_measurements = wait_count_betwen_measurements
        self._freq_minimal_change = freq_minimal_change
        self._angle_change_step = angle_change_step
        self._angle_limit = angle_limit
        self._retreat_steps = retreat_steps

        self._state = States.FIND_CORNER
        self._find_corner_op = FindCornerOp.MOVE_HORISONTAL
        self._find_corner_wait_count = 0

        self._retreat_step_num = 0

        self._finde_side_op = FindSideOp.ROTATE
        self._find_side_wait_count = 0

        self._corner_level = None
        self._left_side = None
        self._right_side = None

    def reset(self):
        self._state = States.FIND_CORNER
        self._find_corner_op = FindCornerOp.MOVE_HORISONTAL
        self._find_corner_wait_count = 0

        self._finde_side_op = FindSideOp.ROTATE
        self._find_side_wait_count = 0

        self._retreat_step_num = 0

        self._corner_level = None
        self._left_side = None
        self._right_side = None

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
            case States.PAUSE:
                return ActionSpace.DO_NOTHING.value
            case States.DONE:
                return ActionSpace.END_EPISODE.value
            case _:
                ValueError("Unknown state")

    def _detect_touch(self, prev_observation, observation) -> bool:
        freq_change_change = observation[1] - prev_observation[1]
        # Если частота упала больше чем на _freq_minimal_change, значит мы коснулись резонатора
        return freq_change_change < -self._freq_minimal_change
            
    def _find_cornet_step(self, prev_observation, observation):
        if self._detect_touch(prev_observation, observation):
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
                if self._detect_touch(prev_observation, observation):
                    side_data = {'angle': observation[4], 'offset': observation[0]}
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
                        zero_side_data = {'angle': 0.0, 'offset': observation[0]}
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
            self._state = States.PAUSE
            return ActionSpace.MOVE_HORIZONTAL.value

    def render_callback(self, canvas, world_transform: Affine2D, ct: CoordinateTransformer):
        def draw_line(linedata, color):
            k, b = math.atan(math.radians(linedata['angle'])), linedata['offset']
            start_pos = world_transform.transform(
                ct.wrap_from_workzone_relative_to_model(WorkzoneRelativeCoordinates(-1.0, k * -1.0 + b)).tuple())
            end_pos = world_transform.transform(
                ct.wrap_from_workzone_relative_to_model(WorkzoneRelativeCoordinates(1.0, k * 1.0 + b)).tuple())
            pygame.draw.line(canvas, start_pos=start_pos, end_pos=end_pos, width=1,  # type: ignore
                             color=color)

        pygame.Surface.lock(canvas)

        #if self._corner_level is not None:
        #    draw_line({'k': 0, 'b': self._corner_level}, (0, 0, 255, 100))

        if self._right_side is not None:
            draw_line(self._right_side, (0, 255, 32))

        if self._left_side is not None:
            draw_line(self._left_side, (0, 255, 32))

        pygame.Surface.unlock(canvas)
