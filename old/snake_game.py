#!/usr/bin/env python

import numpy as np

import matplotlib.pyplot as plt

from models.movement import Movment

if __name__ == '__main__':
    INITILE_POSITION = (0, 1)
    MODEL_TIME_STEP = 0.01
    TAIL_LEN = 10
    
    gen_speed = lambda: np.random.normal(100, 30)
    
    movment = Movment()
    f, ax = plt.subplots()
        
    tail = [ax.plot(INITILE_POSITION[0], INITILE_POSITION[1], 'o-')[0] for _ in range(TAIL_LEN)]
    
    current_pos = INITILE_POSITION
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    
    plt.show(block=False)
    
    while True:
        click = plt.ginput(1, timeout=1)
        if not click:
            continue
        
        new_position = click[0]
        
        path = movment.interpolate_move(current_pos, new_position, gen_speed(), MODEL_TIME_STEP)
        current_pos = new_position
        
        head, = ax.plot(*path[:2], 'o-', color=(1.0, 0.0, 0.0))
        tail.append(head)
        l = tail.pop(0)
        l.remove()
        
        for i, curve in enumerate(tail):
            curve.set_color((1.0 / TAIL_LEN * i, 0, 0))
        
        plt.pause(0.01)
        plt.draw()
