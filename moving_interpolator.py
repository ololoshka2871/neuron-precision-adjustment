#!/usr/bin/env python

import math

from enum import Enum

class MotionStatus(Enum):
    IDLE = 0
    INTERPOLATING = 1


class Command:
    def __init__(self, destinanton: tuple[float, float], F: float, S: float):
        """
        Команда исмуляции движения лазера
        :param destinanton: Координаты точки назначения
        :param F: Скорость перемещения [mm/min]
        :param S: Доля мощности лазера [0..1]
        """
        self.destinanton = destinanton
        self.F = F
        self.S = S


class MovingInterpolator:
    def __init__(self) -> None:
        self._status = MotionStatus.IDLE
        self._is_move_first_interpolation = True

        self._current_startnanos = 0
        self._current_endnanos = 0

        self._now = 0

        self._current_from_x = 0.0
        self._current_from_y = 0.0
        self._current_distance_x = 0.0
        self._current_distance_y = 0.0
        self._current_to_x = 0.0
        self._current_to_y = 0.0
        self._current_cmd_x = 0.0
        self._current_cmd_y = 0.0
        self._current_f = 100.0
        self._current_s = 0.0
        self._current_duration = 0.0

    def begin(self):
        self.current_from_x = 0.0
        self.current_from_y = 0.0
        self.current_to_x = 0.0
        self.current_to_y = 0.0

    def is_busy(self):
        """
        :return: True if the interpolator is busy, False otherwise.
        """
        return self._status == MotionStatus.INTERPOLATING

    def process(self, command: Command):
        if self._status == MotionStatus.IDLE:
            self._current_s = command.S
            self._current_f = command.F
            self._current_to_x, self._current_to_y = command.destinanton

            self._status = MotionStatus.INTERPOLATING
        else:
            raise RuntimeError("Cannot process command while interpolating.")

    def tick(self, now: float) -> MotionStatus:
        """
        Update the interpolator with the current time.
        """
        self._now = int(now * 1_000_000_000.0)

        if self._status == MotionStatus.INTERPOLATING:
            self._interpolate_move()
        
        return self._status
    
    def _interpolate_move(self) -> bool:
        if self._is_move_first_interpolation:
            self._current_distance_x = self._current_to_x - self._current_from_x
            self._current_distance_y = self._current_to_y - self._current_from_y

            self._current_duration = MovingInterpolator.calculate_move_length_nanos(
                self._current_distance_x,
                self._current_distance_y,
                self._current_f,
            )

            self._current_startnanos = self._now
            self._current_endnanos = self._now + int(self._current_duration)
            self._is_move_first_interpolation = False

        # Actual interpolation
        if self._now >= self._current_endnanos:
            # done interpolating
            self._current_from_x = self._current_to_x
            self._current_from_y = self._current_to_y
            self._current_cmd_x = self._current_to_x
            self._current_cmd_y = self._current_to_y
            self._status = MotionStatus.IDLE
            self._is_move_first_interpolation = True
            return self._now == self._current_endnanos
        else:
            fraction_of_move = (self._now - self._current_startnanos) / self._current_duration
            self._current_cmd_x = self._current_from_x + (self._current_distance_x * fraction_of_move);
            self._current_cmd_y = self._current_from_y + (self._current_distance_y * fraction_of_move);
            return True
        
    @staticmethod
    def calculate_move_length_nanos(xdist: float, ydist: float, move_velocity: float) -> float:
        """
        Calculate the time it takes to move a certain distance at a certain velocity.
        :param xdist: The distance to move in the x direction.
        :param ydist: The distance to move in the y direction.
        :param move_velocity: The velocity to move at.
        :return: The time it takes to move the distance at the velocity in nanoseconds.
        """
        length_of_move = math.sqrt(xdist * xdist + ydist * ydist)
        return 1_000_000_000.0 * length_of_move / (move_velocity / 60.0)
    
    def current_position(self) -> tuple[float, float]:
        """
        :return: The current position of the interpolator.
        """
        return self._current_cmd_x, self._current_cmd_y
    
    def from_pos(self) -> tuple[float, float]:
        """
        :return: The starting position of the interpolator.
        """
        return self._current_from_x, self._current_from_y
    
    def to_pos(self) -> tuple[float, float]:
        """
        :return: The ending position of the interpolator.
        """
        return self._current_to_x, self._current_to_y
    
    def current_s(self) -> float:
        """
        :return: The current S value of the interpolator.
        """
        return self._current_s
        

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    interpolator = MovingInterpolator()

    f, ax = plt.subplots(1, 1)

    move_src, = ax.plot([0.0], [0.0], 'ro')
    move_dst, = ax.plot([0.0], [0.0], 'bo')
    current_pos, = ax.plot([0.0], [0.0], 'go')

    move_trace, = ax.plot([0.0, 0.0], [0.0, 0.0], ':', alpha=0.5)

    ax.set_xlim(-10.0, 10.0)
    ax.set_ylim(-10.0, 10.0)

    plt.show(block=False)

    while True:
        status = interpolator.tick(time.time())
        if status == MotionStatus.IDLE:
            click = f.ginput(show_clicks=True, timeout=0.5)
            if len(click) > 0:
                click = click[0]
                interpolator.process(Command(click, 1000.0, 1.0))

        from_pos = interpolator.from_pos()
        dest_pos = interpolator.to_pos()
        cur_pos = interpolator.current_position()

        move_src.set_data(from_pos)
        move_dst.set_data(dest_pos)
        current_pos.set_data(cur_pos)

        move_trace.set_data([from_pos[0], dest_pos[0]], [from_pos[1], dest_pos[1]])

        f.canvas.draw()
        f.canvas.flush_events()

        