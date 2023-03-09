import math

import pytest

from models.movement import Movment

# Создать тесты для проверки методов класс Movment
class TestMovement:
    @pytest.mark.parametrize("src, dst, speed_mm_min, time", [
        ((0, 0), (1, 0), 60, 1),
        ((0, 0), (1, 1), 60, math.sqrt(2)),
        ((0, 0), (1, 1), 120, math.sqrt(2) / 2),
        ((1, 1), (0, 0), 1, math.sqrt(2) * 60),
        ((0, 0), (0, 0), 1, 0),
    ])
    def test_get_move_time(self, src, dst, speed_mm_min, time):
        m = Movment()
        assert m.get_move_time(src, dst, speed_mm_min) == time
        
    @pytest.mark.parametrize("src, dest, speed_mm_min, time_step_s, points_xy", [
        ((0, 0), (1, 0), 60, 1, ([0, 1], [0, 0])),
        ((0, 0), (1, 1), 60, 1, ([0, 1], [0, 1])),
        ((0, 0), (0, 0), 1, 1, ([0], [0]))
    ])
    def test_interpolate_move(self, src, dest, speed_mm_min, time_step_s, points_xy):
        m = Movment()
        assert m.interpolate_move(src, dest, speed_mm_min, time_step_s) == points_xy