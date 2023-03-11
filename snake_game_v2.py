#!/usr/bin/env python


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from misc.common import Rezonator, draw_polygon
from models.movement import Movment
from misc.coordinate_transformer import CoordinateTransformer


def create_tail(ax, tail_len, init_pos):
    return [ax.plot(*init_pos, 'o-')[0] for _ in range(tail_len)]


def draw_rezonators(ax: list[Axes], rez: Rezonator, ct: CoordinateTransformer):
    # базовая точка - середина в месте крепления (0, 0)
    rezonator = rez['rezonator']
    
    # первая ветка
    target1 = rez['targets'][0]
    
    # вторая ветка
    target2 = rez['targets'][1]
    
    # рабочая область
    working_area = rez['working_area']
    
    # Запрещенная область
    forbidden_area = rez['forbidden_area']
    
    #---------------------------
    
    ax[1].set_title('Модель')
    
    real_rezonator = ct.array_wrap_from_model_to_real(rezonator)
    real_target1 = ct.array_wrap_from_model_to_real(target1)
    real_target2 = ct.array_wrap_from_model_to_real(target2)
    real_forbidden_area = ct.array_wrap_from_model_to_real(forbidden_area)
    real_working_area = ct.get_real_working_zone(working_area)
    
    # рисуем базовую форму
    draw_polygon(ax[2], real_rezonator, facecolor='none', edgecolor='black')
    draw_polygon(ax[2], real_target1, color='black')
    draw_polygon(ax[2], real_target2, color='black')
    
    # рисуем рабочую область
    draw_polygon(ax[2], real_working_area, facecolor='none', edgecolor='blue')
    
    # рисуем запрещенную область
    draw_polygon(ax[2], real_forbidden_area, color='magenta')
    
    ax[2].set_xlim(ct.workzone_center[0] - rez.work_zone_size[0] / 2.0, ct.workzone_center[0] + rez.work_zone_size[0] / 2.0)
    ax[2].set_ylim(ct.model_base_point[1], ct.workzone_center[1] + rez.work_zone_size[1] / 2.0)
    
    #---------------------------
    
    model_rezonator = ct.array_wrap_from_real_to_model(real_rezonator)
    model_target1 = ct.array_wrap_from_real_to_model(real_target1)
    model_target2 = ct.array_wrap_from_real_to_model(real_target2)
    model_forbidden_area = ct.array_wrap_from_real_to_model(real_forbidden_area)
    model_working_area = ct.array_wrap_from_real_to_model(real_working_area)
    
    # рисуем базовую форму
    draw_polygon(ax[1], model_rezonator, facecolor='none', edgecolor='black')
    draw_polygon(ax[1], model_target1, color='black')
    draw_polygon(ax[1], model_target2, color='black')

    # рисуем рабочую область
    draw_polygon(ax[1], model_working_area, facecolor='none', edgecolor='blue')

    # рисуем запрещенную область
    draw_polygon(ax[1], model_forbidden_area, color='magenta')
    
    ax[1].set_xlim(min(model_working_area[:, 0]), max(model_working_area[:, 0]))
    ax[1].set_ylim(min(model_working_area[:, 1]), max(model_working_area[:, 1]))

    #---------------------------

    ax[0].set_title('Рабочая зона')
    
    wz_rezonator = ct.array_wrap_from_real_to_workzone(real_rezonator)
    wz_target1 = ct.array_wrap_from_real_to_workzone(real_target1)
    wz_target2 = ct.array_wrap_from_real_to_workzone(real_target2)
    wz_forbidden_area = ct.array_wrap_from_real_to_workzone(real_forbidden_area)
    wz_working_area = ct.array_wrap_from_real_to_workzone(real_working_area)
    
    # рисуем базовую форму
    draw_polygon(ax[0], wz_rezonator, facecolor='none', edgecolor='black')
    draw_polygon(ax[0], wz_target1, color='black')
    draw_polygon(ax[0], wz_target2, color='black')

    # рисуем рабочую область
    draw_polygon(ax[0], wz_working_area, facecolor='none', edgecolor='blue')

    # рисуем запрещенную область
    draw_polygon(ax[0], wz_forbidden_area, color='magenta')
    
    ax[0].set_xlim(-1, 1)
    ax[0].set_ylim(-1, 1)
    
    
def add_tail_segment(ax: Axes, path, tail: list[Line2D], color):
    head, = ax.plot(*path, 'o-', color=color)
    tail.append(head)
    lp = tail.pop(0)
    lp.remove()
    

def update_tail_colors(tail: list[Line2D]):
    tail_len = len(tail)
    for i, line in enumerate(tail):
        line.set_color((1.0 / tail_len * i, 0, 0))
    

if __name__ == '__main__':
    INITILE_POSITION = (0, 1)
    MODEL_TIME_STEP = 0.01
    TAIL_LEN = 10
    
    gen_speed = lambda: np.random.normal(100, 30)
    
    offset = (np.random.random() * 0.3, np.random.random() * 0.5)
    angle = np.random.random() * 20 - 10
    print('offset: {}, angle: {}'.format(offset, angle))
    
    rez = Rezonator.load()
    
    ct = CoordinateTransformer(rez, workzone_center=(15.3, 7.62), offset=offset, angle=angle)
    movment = Movment()
    f, ax = plt.subplots(1 ,3)
    
    draw_rezonators(ax, rez, ct)
    
    current_pos_wz = INITILE_POSITION
    current_pos_model = ct.array_wrap_from_workzone_to_model(np.array(INITILE_POSITION))
    current_pos_real = ct.array_wrap_from_workzone_to_real(np.array(INITILE_POSITION))
    
    wz_tail = create_tail(ax[0], TAIL_LEN, INITILE_POSITION)
    model_tail = create_tail(ax[1], TAIL_LEN, current_pos_model)
    real_tail = create_tail(ax[2], TAIL_LEN, current_pos_real)
    
    plt.show(block=False)
    
    while True:
        click = plt.ginput(1, timeout=1)
        if not click:
            continue
        
        new_position = click[0]
        if min(new_position) < -1 or max(new_position) > 1:
            continue
        
        speed = gen_speed()
        path_wz = movment.interpolate_move(current_pos_wz, new_position, gen_speed(), MODEL_TIME_STEP)
        path_model = ct.array_wrap_from_workzone_to_model(np.array(path_wz[:2]).T).T
        path_real = ct.array_wrap_from_workzone_to_real(np.array(path_wz[:2]).T).T
        
        current_pos_wz = new_position
        current_pos_model = path_model[-1]
        current_pos_real = path_real[-1]
        
        color = (1.0, 0.0, 0.0)
        add_tail_segment(ax[0], path_wz[:2], wz_tail, color)
        add_tail_segment(ax[1], path_model[:2], model_tail, color)
        add_tail_segment(ax[2], path_real[:2], real_tail, color)
        
        update_tail_colors(wz_tail)
        update_tail_colors(model_tail)
        update_tail_colors(real_tail)
        
        plt.pause(0.01)
        plt.draw()
