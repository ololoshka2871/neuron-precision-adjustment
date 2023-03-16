#!/usr/bin/env python

import math

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class AdjustZoneModel:
    METAL_DENSITY = 10.49 / math.pow(1e+1, 3)  # g/sm^3 -> g/mm^3
    SENSIVITY_BASE = 3e-9  # g/Hz

    def __init__(self,
                 size: tuple[float, float],
                 divizion: tuple[int, int],
                 layer_thikness=0.5e-6 * 1e3,
                 sensitivity_multiplicator=lambda pos: 1.0,
                 power_threshold=0.0,
                 energy_consume_rate=lambda pos: 1.0):
        """
        Создать новый экземпляр модели мишени
        :param size: физический размер мишени в милиметрах
        :param divizion: количество чанков по ширине и высоте на  которое разбивается вся мишень
        :param layer_thiknes: толщина слоя серебра на мишени, мм
        :param sensitivity_multiplicator: функция, которая возвращает коэффициент чувствительности в зависимости от позиции, позиция в пределах 0.0..1.0
        :param power_threshold: минимальная мощность лазера, необходимая для начала испарения серебра в пределах 0.0..1.0
        :param energy_consume_rate: функция, которая возвращает коэффициент поглащения энергии в зависимости количества серебра в данном месте, количество серебра в пределах 0.0..1.0
        """

        full_mass = size[0] * size[1] * \
            layer_thikness * AdjustZoneModel.METAL_DENSITY

        self.chank_freq_offset_max = full_mass / \
            AdjustZoneModel.SENSIVITY_BASE / (divizion[0] * divizion[1])
        self.field = np.array([[1.0 for y in range(divizion[1])]
                              for x in range(divizion[0])])
        self.sensitivity_multiplicator = sensitivity_multiplicator
        self.power_threshold = power_threshold
        self.energy_consume_rate = energy_consume_rate

        self.full_freq_offset = self._current_freq_offset()

    def _color(self, x, y) -> tuple[float, float, float]:
        w, h = self.field.shape
        c = float(self.field[x][y])
        s = self.sensitivity_multiplicator((x / w, y / h))
        return (c, 0.5, s)

    def to_grid(self, base_pos=(0.0, 0.0), size=(1.0, 1.0)):
        w, h = self.field.shape
        size = (size[0] / w, size[1] / h)

        return [[patches.Rectangle((base_pos[0] + x * size[0], base_pos[1] + y * size[1]), size[0], size[1],
                                   linewidth=1, edgecolor=(0, 0, 0), facecolor=self._color(x, y))
                for y in range(h)] for x in range(w)]
    
    def to_color_map(self) -> list[list[tuple[float, float, float]]]:
        w, h = self.field.shape
        return [[self._color(x, y) for y in range(h)] for x in range(w)]

    def find_chank(self, pos):
        """
        Функция возвращает чанк, в котором находится точка
        :param pos: координаты точки в пределах 0.0..1.0
        :return: координаты чанка
        """
        w, h = self.field.shape
        return (int(pos[0] * w), int(pos[1] * h))

    def update(self, pos, time: float, power=1.0) -> bool:
        """
        Обновить состояние мишени
        :param pos: координаты точки в пределах 0.0..1.0
        :param time: время в секундах
        :param power: мощность лазера [W]
        :return: True, если мишень еще не испарилась полностью
        """
        chank_pos = self.find_chank(pos)
        current_value = self.field[chank_pos]

        if current_value > 0 and power > self.power_threshold:
            current_value = max(0, current_value -
                                self.energy_consume_rate(pos) * time * power)
            self.field[chank_pos] = current_value
            return current_value > 0
        
        return False

    def _current_freq_offset(self):
        size = self.field.shape

        summ = 0
        for x, row in enumerate(self.field):
            for y, chank_mass in enumerate(row):
                multiplicator = self.sensitivity_multiplicator(
                    (x / size[0], y / size[1]))
                summ += chank_mass * self.chank_freq_offset_max * multiplicator
        return summ

    def freq_change(self) -> float:
        return self.full_freq_offset - self._current_freq_offset()
    
    def max_adjustment(self) -> float:
        """
        Возвращает таксимально-возможное изменение частоты резонатора если испарить все серебро.
        """
        return self.full_freq_offset


def draw_model(axis: plt.Axes, rects: list[list]) -> list[list[patches.Patch]]:
    return [[axis.add_patch(rect) for rect in row] for row in rects]


def update_target(position: tuple, target_polygon_real, target_polygon_original, zone: AdjustZoneModel,
                  patches: list[list[patches.Patch]], untransformed_pos, target_size, **update_args) -> bool:
    if is_point_inside_polygon(position, target_polygon_real):
        original_base_pos = target_polygon_original[0]
        target_pos = (untransformed_pos - original_base_pos) / target_size
        d = zone.update(target_pos, **update_args)
        if not d:
            print(f"Target {target_pos} deepleed!")

        for row in patches:
            for patch in row:
                patch.remove()

        rects = zone.to_grid(original_base_pos, original_target_size)
        patches[:] = draw_model(untr_pg, rects)
        return True
    return False


def create_linear_sensivity_multiplicator(min, max):
    def sensivity(pos):
        pos = pos[1]
        if pos < 0:
            return min
        elif pos > max:
            return max
        else:
            return (max - min) * pos + min
    return sensivity


if __name__ == '__main__':
    from misc.common import draw_polygon, transform_all
    from misc.inside_detector import is_point_inside_polygon
    from old.generate_playground import generate_playground

    f, ax = plt.subplots(1, 2)

    pg = ax[0]

    # Генерируем случайное смещение и случайный угол поворота
    offset = (np.random.random() * 0.3, np.random.random() * 0.5)
    angle = np.random.random() * 20 - 10
    print('offset: {}, angle: {}'.format(offset, angle))

    playground = generate_playground(offset=offset, angle=angle)

    current_pos_global = [0, playground['working_area'][2][1]]

    # рисуем базовую форму
    draw_polygon(pg, playground['rezonator'],
                 edgecolor='black', facecolor='none')

    # рисуем цель
    draw_polygon(pg, playground['targets'][0], color='green')
    draw_polygon(pg, playground['targets'][1], color='green')

    # рисуем запрещенную область
    draw_polygon(pg, playground['forbidden_area'], color='magenta')

    # рисуем рабочую область
    draw_polygon(pg, playground['working_area'],
                 edgecolor='blue', facecolor='none')

    # текущая точка
    current_pos, = pg.plot(current_pos_global[0], current_pos_global[1], 'ro')

    # Установка границ по осям X и Y чтобы видно было только рабочую область
    pg.set_xlim(playground['working_area'][0][0]-0.1,
                playground['working_area'][1][0]+0.1)
    pg.set_ylim(playground['working_area'][1][1]-0.1,
                playground['working_area'][2][1]+0.1)

    # ------------------------------------------------------------

    # точность разбиения цели на чанки
    CHANK_GRID_PRECISION_X = 3
    CHANK_GRID_PRECISION_Y = 20

    original_target = playground['original']['targets'][0]
    original_target_size = original_target[2] - original_target[0]
    target_size_ratio = original_target_size[0] / original_target_size[1]

    inverted_transform_matrix = playground['transform_matrix'].inverted()

    untr_pg = ax[1]

    original_rezonator = playground['original']['rezonator']
    original_forbidden_area = playground['original']['forbidden_area']
    un_transformed_work_zone = transform_all(
        [playground['original']['working_area']], inverted_transform_matrix)[0]

    draw_polygon(untr_pg, original_rezonator,
                 edgecolor='black', facecolor='none')
    draw_polygon(untr_pg, original_forbidden_area, color='magenta')
    draw_polygon(untr_pg, un_transformed_work_zone,
                 edgecolor='blue', facecolor='none')

    f_sensivity = create_linear_sensivity_multiplicator(0.5, 1.0)

    zone1 = AdjustZoneModel(size=original_target_size, divizion=(
        CHANK_GRID_PRECISION_X, CHANK_GRID_PRECISION_Y), sensitivity_multiplicator=f_sensivity)
    rects = zone1.to_grid(
        playground['original']['targets'][0][0], original_target_size)
    patches1 = draw_model(untr_pg, rects)

    zone2 = AdjustZoneModel(size=original_target_size, divizion=(
        CHANK_GRID_PRECISION_X, CHANK_GRID_PRECISION_Y), sensitivity_multiplicator=f_sensivity)
    rects = zone2.to_grid(
        playground['original']['targets'][1][0], original_target_size)
    patches2 = draw_model(untr_pg, rects)

    # текущая точка
    transformed_pos = transform_all(
        [current_pos_global], inverted_transform_matrix)[0]
    current_pos_transformed, = untr_pg.plot(
        transformed_pos[0], transformed_pos[1], 'ro')

    untr_pg.set_xlim(
        un_transformed_work_zone[0][0]-0.1, un_transformed_work_zone[2][0]+0.1)
    untr_pg.set_ylim(
        un_transformed_work_zone[1][1]-0.1, un_transformed_work_zone[3][1]+0.1)

    plt.show(block=False)

    time = 0.1

    while True:
        click = f.ginput(show_clicks=True)
        if len(click) == 0:
            exit(0)
        else:
            click = click[0]

        current_pos.set_data(click[0], click[1])

        transformed_pos = transform_all([click], inverted_transform_matrix)[0]
        current_pos_transformed.set_data(
            transformed_pos[0], transformed_pos[1])

        if not update_target(click, playground['targets'][0], playground['original']['targets'][0], zone1, patches1,
                             transformed_pos, original_target_size, time=time):
            if not update_target(click, playground['targets'][1], playground['original']['targets'][1], zone2, patches2,
                                 transformed_pos, original_target_size, time=time):
                plt.draw()
                continue

        ch = [zone1.freq_change(), zone2.freq_change()]
        total_change = sum(ch)
        diff = (ch[0] - ch[1]) / total_change
        print('freq_changed: {} ({} + {}), diff: {}%'.format(total_change,
              ch[0], ch[1], diff * 100))

        plt.draw()
