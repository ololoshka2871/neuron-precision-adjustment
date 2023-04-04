from enum import Enum
import numpy as np
import math

from gym_quarz.envs.quartz6 import ActionSpace

class States(Enum):
    FIND_CORNER = 0 # поиск торчащего уголка резонатора, просто змейкой движемся вниз пока частота не изменится. 
                    # Ждем после заждого прохода чтобы произошло измерение
    RETREAT = 1     # отступаем от уголка, чтобы не было перекрытия сигналов

    DONE = 99 # конец

class FindCornerOp(Enum):
    MOVE_HORISONTAL = 0
    MOVE_WAIT = 1
    MOVE_DOWN = 2


class AlgorithmicController:
    """
    Контроллер алгоритмический
    """

    def __init__(self,
                 wait_count_betwen_measurements: int = 4,
                 freq_minimal_change: float = 0.1,):
        self._wait_count_betwen_measurements = wait_count_betwen_measurements
        self._freq_minimal_change = freq_minimal_change

        self._state = States.FIND_CORNER
        self._find_corner_op = FindCornerOp.MOVE_HORISONTAL
        self._find_corner_wait_count = 0

    def reset(self):
        self._state = States.FIND_CORNER
        self._find_corner_op = FindCornerOp.MOVE_HORISONTAL
        self._find_corner_wait_count = 0

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
        if freq_change_change < -self._freq_minimal_change: # Если частота упала больше чем на _freq_minimal_change, значит мы коснулись уголка
            self._state = States.RETREAT
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
