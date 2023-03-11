import numpy as np
import matplotlib.pyplot as plt

from misc.common import draw_polygon, Rezonator
from misc.coordinate_transformer import RealCoordinates, ModelCoordinates, WorkzoneRelativeCoordinates, CoordinateTransformer


if __name__ == "__main__":
    from misc.common import draw_polygon
    
    f, axis = plt.subplots(1, 3)

    rez = Rezonator.load()
    
    offset = (0, 0)  # (np.random.random() * 0.3, np.random.random() * 0.5)
    angle = 10  # np.random.random() * 20 - 10
    print('offset: {}, angle: {}'.format(offset, angle))
    
    ct = CoordinateTransformer(rez, workzone_center=(24, 8), offset=offset, angle=angle)
    
    #---------------------------
    
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

    ax = axis[0]
    ax.set_title('Реальность')
    
    real_rezonator = ct.array_wrap_from_model_to_real(rezonator)
    real_target1 = ct.array_wrap_from_model_to_real(target1)
    real_target2 = ct.array_wrap_from_model_to_real(target2)
    real_forbidden_area = ct.array_wrap_from_model_to_real(forbidden_area)
    real_working_area = ct.get_real_working_zone(working_area)
    
    # рисуем базовую форму
    draw_polygon(ax, real_rezonator, facecolor='none', edgecolor='black')
    draw_polygon(ax, real_target1, color='black')
    draw_polygon(ax, real_target2, color='black')
    
    # рисуем рабочую область
    draw_polygon(ax, real_working_area, facecolor='none', edgecolor='blue')
    
    # рисуем запрещенную область
    draw_polygon(ax, real_forbidden_area, color='magenta')
    
    ax.set_xlim(ct.workzone_center[0] - rez.work_zone_size[0] / 2.0, ct.workzone_center[0] + rez.work_zone_size[0] / 2.0)
    ax.set_ylim(ct.model_base_point[1], ct.workzone_center[1] + rez.work_zone_size[1] / 2.0)
        
    #---------------------------
    
    ax = axis[1]
    ax.set_title('Модель')
    
    model_rezonator = ct.array_wrap_from_real_to_model(real_rezonator)
    model_target1 = ct.array_wrap_from_real_to_model(real_target1)
    model_target2 = ct.array_wrap_from_real_to_model(real_target2)
    model_forbidden_area = ct.array_wrap_from_real_to_model(real_forbidden_area)
    model_working_area = ct.array_wrap_from_real_to_model(real_working_area)
    
    # рисуем базовую форму
    draw_polygon(ax, model_rezonator, facecolor='none', edgecolor='black')
    draw_polygon(ax, model_target1, color='black')
    draw_polygon(ax, model_target2, color='black')

    # рисуем рабочую область
    draw_polygon(ax, model_working_area, facecolor='none', edgecolor='blue')

    # рисуем запрещенную область
    draw_polygon(ax, model_forbidden_area, color='magenta')
    
    ax.set_xlim(min(model_working_area[-2:, 0]), max(model_working_area[1:3, 0]))
    ax.set_ylim(min(model_working_area[1:3, 1]), max(model_working_area[-2:, 1]))

    #---------------------------

    ax = axis[2]
    ax.set_title('Рабочая зона')
    
    wz_rezonator = ct.array_wrap_from_real_to_workzone(real_rezonator)
    wz_target1 = ct.array_wrap_from_real_to_workzone(real_target1)
    wz_target2 = ct.array_wrap_from_real_to_workzone(real_target2)
    wz_forbidden_area = ct.array_wrap_from_real_to_workzone(real_forbidden_area)
    wz_working_area = ct.array_wrap_from_real_to_workzone(real_working_area)
    
    # рисуем базовую форму
    draw_polygon(ax, wz_rezonator, facecolor='none', edgecolor='black')
    draw_polygon(ax, wz_target1, color='black')
    draw_polygon(ax, wz_target2, color='black')

    # рисуем рабочую область
    draw_polygon(ax, wz_working_area, facecolor='none', edgecolor='blue')

    # рисуем запрещенную область
    draw_polygon(ax, wz_forbidden_area, color='magenta')
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    #---------------------------

    ## Установка одинакового масштаба по осям X и Y
    #plt.axis('equal')

    plt.show()
