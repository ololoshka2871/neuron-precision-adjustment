from enum import Enum
from matplotlib.transforms import Affine2D
import numpy as np
import math

import pygame

from gym_quarz.envs.quartz6 import ActionSpace
from misc.coordinate_transformer import CoordinateTransformer, WorkzoneRelativeCoordinates


class States(Enum):
    # поиск торчащего уголка резонатора, просто змейкой движемся вниз пока частота не изменится.
    FIND_CORNER = 0
    # Ждем после заждого прохода чтобы произошло измерение
    RETREAT = 1     # отступаем от уголка, чтобы не было перекрытия сигналов

    DONE = 99  # конец


class FindCornerOp(Enum):
    MOVE_HORISONTAL = 0
    MOVE_WAIT = 1
    MOVE_DOWN = 2


class AlgorithmicController:
    """
    Контроллер алгоритмический
    """

    def __init__(self,
                 angle_change_step: float,
                 angle_limit: float,
                 wait_count_betwen_measurements: int = 4,
                 freq_minimal_change: float = 0.1,):
        self._wait_count_betwen_measurements = wait_count_betwen_measurements
        self._freq_minimal_change = freq_minimal_change
        self._angle_change_step = angle_change_step
        self._angle_limit = angle_limit

        self._state = States.FIND_CORNER
        self._find_corner_op = FindCornerOp.MOVE_HORISONTAL
        self._find_corner_wait_count = 0

        self._corner_level = None

    def reset(self):
        self._state = States.FIND_CORNER
        self._find_corner_op = FindCornerOp.MOVE_HORISONTAL
        self._find_corner_wait_count = 0

        self._corner_level = None

    def sample_action(self, prev_observation, observation):
        match self._state:
            case States.FIND_CORNER:
                return self._find_cornet_step(prev_observation, observation)
            case States.RETREAT:
                return self._retreat_step(prev_observation, observation)
            case States.DONE:
                return 4
            case _:
                ValueError("Unknown state")

    def _find_cornet_step(self, prev_observation, observation):
        freq_change_change = observation[1] - prev_observation[1]
        # Если частота упала больше чем на _freq_minimal_change, значит мы коснулись уголка
        if freq_change_change < -self._freq_minimal_change:
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

        self._state = States.DONE
        return 2

    def render_callback(self, canvas, world_transform: Affine2D, ct: CoordinateTransformer):
        pygame.Surface.lock(canvas)

        if self._corner_level is not None:
            start_pos = world_transform.transform(
                ct.wrap_from_workzone_relative_to_model(WorkzoneRelativeCoordinates(-1, self._corner_level)).tuple())
            end_pos = world_transform.transform(
                ct.wrap_from_workzone_relative_to_model(WorkzoneRelativeCoordinates(1, self._corner_level)).tuple())
            pygame.draw.line(canvas, start_pos=start_pos, end_pos=end_pos, width=1, # type: ignore
                             color=(0, 0, 255, 100))  

        pygame.Surface.unlock(canvas)
