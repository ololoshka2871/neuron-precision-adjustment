#!/usr/bin/env python

from enum import Enum

import numpy as np

from .adjust_zone_model import AdjustZoneModel, create_linear_sensivity_multiplicator
from misc.common import Rezonator, build_transform_matrix
from .temperature_model import TemperatureModel, QUARTZ_DESTENSITY, QUARTZ_HEAT_CAPACITY
from misc.inside_detector import is_point_inside_polygon


class Zone(Enum):
    """
    Зоны резонатора
    """
    TARGET1 = 0  # Мишень 1
    TARGET2 = 1  # Мишень 2
    FORBIDDEN = 2  # Запрещенная зона
    BODY = 3  # Тело резонатора без покрытия
    NONE = 4  # За пределами резонатора


class Playground:
    """
    Игровое поле, содержит резонатор, мишени, запрещенную зону, рабочую зону.
    Трансформирует координаты
    """

    def __init__(self, offset: tuple[float, float], angle: float):
        self._transformation_matrix = build_transform_matrix(offset, angle)
        self._reverse_transformation_matrix = self._transformation_matrix.inverted()

        # Кэши преобразованных объектов
        self._rezonator = None
        self._forbidden_area = None
        self._targets = None

    @property
    def rezonator(self):
        if self._rezonator is None:
            self._rezonator = self._transformation_matrix.transform(
                RezonatorModel.REZONATOR['rezonator'])
        return self._rezonator

    @property
    def forbidden_area(self):
        if self._forbidden_area is None:
            self._forbidden_area = self._transformation_matrix.transform(
                RezonatorModel.REZONATOR['forbidden_area'])
        return self._forbidden_area

    @property
    def working_area(self):
        return RezonatorModel.REZONATOR['working_area']

    def working_area_limits(self, offset: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Возвращает границы рабочей зоны с учетом отступа
        """
        wa = self.working_area
        return (wa[0][0] - offset, wa[1][0] + offset), (wa[1][1] - offset, wa[2][1] + offset)

    def target(self, index: int):
        if self._targets is None:
            self._targets = [self._transformation_matrix.transform(
                t) for t in RezonatorModel.REZONATOR['targets']]
        return self._targets[index]

    def map_to_model(self, obj):
        """
        Преобразует координаты из глобальной системы координат в такую, где резонатор имеет нулевое смещеине и угол
        """
        return self._reverse_transformation_matrix.transform(obj)


class ModelView:
    """
    Состояние резонатора для отображения на экране
    """

    def __init__(self, offset: tuple[float, float], angle: float, zones: list[AdjustZoneModel]):
        self._transformation_matrix = build_transform_matrix(offset, angle)
        self._reverse_transformation_matrix = self._transformation_matrix.inverted()
        self._zones = zones

        # Кэши преобразованных объектов
        self._working_area = None

    @property
    def rezonator(self):
        return RezonatorModel.REZONATOR['rezonator']

    @property
    def forbidden_area(self):
        return RezonatorModel.REZONATOR['forbidden_area']

    @property
    def working_area(self):
        if self._working_area is None:
            self._working_area = self._reverse_transformation_matrix.transform(
                RezonatorModel.REZONATOR['working_area'])
        return self._working_area

    @staticmethod
    def _detect_zone_slow(position:tuple[float, float]) -> Zone:
        """
        Функция определяет зону резонатора в которой находится точка (медленный вариант)
        :param position: координаты точки
        :return: зона
        """
        if is_point_inside_polygon(position, RezonatorModel.REZONATOR['rezonator']):
            if is_point_inside_polygon(position, RezonatorModel.REZONATOR['forbidden_area']):
                return Zone.FORBIDDEN
            elif is_point_inside_polygon(position, RezonatorModel.REZONATOR['targets'][0]):
                return Zone.TARGET1
            elif is_point_inside_polygon(position, RezonatorModel.REZONATOR['targets'][1]):
                return Zone.TARGET2
            else:
                return Zone.BODY
        else:
            return Zone.NONE

    @staticmethod
    def detect_zone(position, hint: Zone = Zone.NONE) -> Zone:
        """
        Функция определяет зону резонатора в которой находится точка. Сначала проверяет зону, указанную в hint, если точка
            не входит в нее, то выполняется полный перебор всех зон.
        :param position: Координаты точки
        :param hint: Предполагаемая зона
        :return: зона
        """

        match hint:
            case Zone.TARGET1:
                if is_point_inside_polygon(position, RezonatorModel.REZONATOR['targets'][0]):
                    return Zone.TARGET1
            case Zone.TARGET2:
                if is_point_inside_polygon(position, RezonatorModel.REZONATOR['targets'][1]):
                    return Zone.TARGET2
            case Zone.FORBIDDEN:
                if is_point_inside_polygon(position, RezonatorModel.REZONATOR['forbidden_area']):
                    return Zone.FORBIDDEN
            case _:
                pass
        return ModelView._detect_zone_slow(position)

    def working_area_limits(self, offset: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Возвращает границы рабочей зоны с учетом отступа
        """
        wa = self.working_area
        return ((min(wa[0][0], wa[3][0]) - offset, max(wa[1][0], wa[2][0]) + offset),
                (min(wa[0][1], wa[1][1]) - offset, max(wa[2][1], wa[3][1]) + offset))

    def map_to_view(self, pos: tuple[float, float]):
        """
        Преобразует координаты из локальной системы координат резонатора в глобальную
        """
        return self._transformation_matrix.transform(pos)

    def target(self, index: int):
        return self._zones[index].to_grid(
            RezonatorModel.REZONATOR.get_target_base_point(index),
            RezonatorModel.REZONATOR.target_zone_size)

    def target_color_map(self, index: int):
        return self._zones[index].to_color_map()

    @staticmethod
    def map_to_zone(zone: Zone, model_pos) -> tuple[float, float]:
        match zone:
            case Zone.TARGET1:
                base = RezonatorModel.REZONATOR.get_target_base_point(0)
            case Zone.TARGET2:
                base = RezonatorModel.REZONATOR.get_target_base_point(1)
            case _:
                raise ValueError(f"Zone {zone} is not supported")

        size = RezonatorModel.REZONATOR.target_zone_size
        return ((model_pos[0] - base[0]) / size[0], (model_pos[1] - base[1]) / size[1])


class Metrics(dict):
    """
    Метрики резонатора
    - static_freq_change - статическое изменение частоты только от испарения серебра (без учета температуры)
    - freq_change: float - общее изменение частоты с учетом температуры
    - freq_change_branches: tuple[float, float] - изменение частоты в 2 ветвях
    - disbalance: float - относительное расхождение частот в 2 ветвях
    - penalty_energy: float - общая энергия переданная в запрещенную зону (штрафные очки)
    - temperature: float - температура резонатора
    """

    def __init__(self, freq_change_branches: tuple[float, float], penalty_energy: float, temperature: float, tfk: float):
        self["freq_change_branches"] = freq_change_branches
        self["penalty_energy"] = penalty_energy
        self["temperature"] = temperature
        self._tfk = tfk

    def __getitem__(self, key: str):
        if key == "static_freq_change":
            return sum(self["freq_change_branches"])  # type: ignore
        elif key == "freq_change":
            return self["static_freq_change"] + self['temperature'] * self._tfk
        elif key == "disbalance":
            total_change = self["static_freq_change"]
            if total_change == 0.0:
                return 0.0
            else:
                return (self["freq_change_branches"][1] - self["freq_change_branches"][0]) / self["static_freq_change"]

        return super().__getitem__(key)


class RezonatorModel:
    """
    Полная модель резонатора, включает 2 мишени, запрещенную зону, и нейтральную зону.
    Учитывает нагрев и охлаждение.
    Регистрация попаданий в запрещенную зону в качестве "штрафных" очков.
    """

    CELSUSS_TO_KELVIN = 273.15
    REZONATOR = Rezonator.load()
    TARGET_CHANK_DIVIZION = (3, 20)

    def __init__(self,
                 rezonator_thickness: float = 0.23,
                 heat_dissipation_rate: float = 0.9,
                 ambient_temperatire: float = 0.0,
                 power_threshold: float = 0.05,
                 tfk: float = -0.05):
        """
        :param rezonator_thickness: Толщина резонатора [мм]
        :param heat_dissipation_rate: Скорость рассеяние энергии в окружающую стреду [0..1]
        :param ambient_temperatire: температура окружающей среды, [градусы Цельсия]
        :param power_threshold: Минимальная доля мощности лазера, необходимая чтобы началось испарение серебра
        :param tfk: температурная чувствительность резонатора Hz/K
        """

        self._tfk = tfk

        def consume_rate(current_temperature: float, zone=Zone.NONE) -> float:
            match zone:
                case Zone.TARGET1 | Zone.TARGET2:
                    return 1.0 - 0.98
                case Zone.FORBIDDEN:
                    return 1.0 - 0.95
                case _:
                    return 1.0 - 0.8

        volume = RezonatorModel.REZONATOR.body_volume(
            rezonator_thickness) * pow(1e-3, 3)  # mm^3 -> m^3
        rezonator_heat_capacity = QUARTZ_HEAT_CAPACITY * QUARTZ_DESTENSITY * volume

        self._temperature_model = TemperatureModel(rezonator_heat_capacity, heat_dissipation_rate,
                                                   ambient_temperature=ambient_temperatire + RezonatorModel.CELSUSS_TO_KELVIN,
                                                   consume_rate=consume_rate)  # type: ignore

        target_zone_size = RezonatorModel.REZONATOR.target_zone_size

        f_sensivity = create_linear_sensivity_multiplicator(0.5, 1.0)
        f_consume = create_linear_sensivity_multiplicator(0.75, 1.0)

        self._adj_zones = [AdjustZoneModel(target_zone_size,
                                           divizion=RezonatorModel.TARGET_CHANK_DIVIZION,
                                           sensitivity_multiplicator=f_sensivity,
                                           power_threshold=power_threshold,
                                           energy_consume_rate=f_consume) for _ in range(2)]

        self._forbidden_zone_energy_accumulated = 0.0

    def get_playground(self, offset: tuple, angle: float) -> Playground:
        """
        Возвращает объект Playground:

        :param offset: смещение резонатора относительно нуля координат
        :param angle: угол поворота резонатора в градусах
        :return: объект Playground, содержащий резонатор, мишени, запрещенную зону, рабочую зону
        """
        return Playground(offset, angle)

    def get_model_view(self, offset: tuple, angle: float) -> ModelView:
        """
        Возвращает словарь состояния резонатора чтобы отобразить его на экране:

        Необходимы следующие сведения о настоящем положении резонатора:
        :param offset: смещение резонатора относительно нуля координат
        :param angle: угол поворота резонатора в градусах
        """
        return ModelView(offset, angle, self._adj_zones)

    def get_metrics(self) -> Metrics:
        """
        Возвращает метриками симуляции резонатора:
        """
        return Metrics((self._adj_zones[0].freq_change(), self._adj_zones[1].freq_change()),
                       self._forbidden_zone_energy_accumulated,
                       self._temperature_model.current_temperature() - RezonatorModel.CELSUSS_TO_KELVIN,
                       self._tfk)

    def heat_body(self, power: float, time: float):
        """
        Обновляет состояние резонатора. Считается, что лазер воздействует на позицию position время time с мощностью power.
        Если обновлять состояние резонатора достаточно часто, то можно считать, что лазер воздействует непрерывно.
        """
        self._temperature_model.tick(power, time, zone=Zone.BODY)

    def heat_forbidden(self, power: float, time: float):
        """
        Поподание в запрещенную зону на время time с мощностью power.
        Если обновлять состояние резонатора достаточно часто, то можно считать, что лазер воздействует непрерывно.
        """
        self._forbidden_zone_energy_accumulated += power * time
        self._temperature_model.tick(power, time, zone=Zone.FORBIDDEN)

    def target(self, zone: Zone, positions: tuple[float, float], power: float, time: float):
        """
        Поподание в мишень на время time с мощностью power.
        :param zone: мишень
        :param positions: относительная позиция на мишени, в которую попадает лазер (0..1)
        :param power: мощность лазера
        :param time: время попадания
        """
        depleeted = self._adj_zones[zone.value].update(positions, time, power)
        self._temperature_model.tick(
            power, time, zone=Zone.BODY if depleeted else zone)

    def idle(self, time: float):
        """
        Резонатор остывает время time.
        """
        self._temperature_model.tick(0.0, time)

    @staticmethod
    def get_working_zone(offset=(0.0, 0.0)) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Возвращает рабочую зону резонатора.
        """
        return offset, RezonatorModel.REZONATOR.target_zone_size
    
    @property
    def possible_freq_adjust(self) -> float:
        """
        Возвращает таксимально-возможное изменение частоты резонатора если испарить все серебро.
        """
        return sum(map(AdjustZoneModel.max_adjustment, self._adj_zones))
    
    def map_local_to_global(self, wz_pos: tuple[float, float]) -> tuple[float, float]:
        """
        Преобразует локальные координаты внутри рабочей зоны в глобальные.
        :param wz_pos: локальные координаты относитеьно центра рабочей зоны
        """
        wz_center_pos = RezonatorModel.REZONATOR.work_zone_center_pos

        return (wz_pos[0] + wz_center_pos[0], wz_pos[1] + wz_center_pos[1])


if __name__ == '__main__':
    import time
    import datetime as dt
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from misc.common import draw_polygon
    from adjust_zone_model import draw_model

    LASER_POWER = 30.0  # [W]

    f, ax = plt.subplots(1, 3)

    rezonator = RezonatorModel()

    # Генерируем случайное смещение и случайный угол поворота
    offset = (np.random.random() * 0.3, np.random.random() * 0.5)
    angle = np.random.random() * 20 - 10
    print('offset: {}, angle: {}'.format(offset, angle))

    # ----------------------------------------

    pg = ax[0]

    playground = rezonator.get_playground(offset, angle)

    current_pos_global = [0, playground.working_area[2][1]]

    # рисуем базовую форму
    draw_polygon(pg, playground.rezonator,
                 edgecolor='black', facecolor='none')

    # рисуем цели
    draw_polygon(pg, playground.target(0), color='green')
    draw_polygon(pg, playground.target(1), color='green')

    # рисуем запрещенную область
    draw_polygon(pg, playground.forbidden_area, color='magenta')

    # рисуем рабочую область
    draw_polygon(pg, playground.working_area,
                 edgecolor='blue', facecolor='none')

    # текущая точка
    current_pos, = pg.plot(current_pos_global[0], current_pos_global[1], 'ro')

    # Установка границ по осям X и Y чтобы видно было только рабочую область
    limits = playground.working_area_limits(0.1)
    pg.set_xlim(*limits[0])
    pg.set_ylim(*limits[1])

    # ----------------------------------------

    mv = ax[1]

    model_view = rezonator.get_model_view(offset, angle)

    # рисуем базовую форму
    draw_polygon(mv, model_view.rezonator,
                 edgecolor='black', facecolor='none')

    # рисуем цели
    patches = [draw_model(mv, model_view.target(i)) for i in range(2)]

    # рисуем запрещенную область
    draw_polygon(mv, model_view.forbidden_area, color='magenta')

    # рисуем рабочую область
    draw_polygon(mv, model_view.working_area,
                 edgecolor='blue', facecolor='none')

    # текущая точка
    model_pos = playground.map_to_model(current_pos_global)
    current_pos_transformed, = mv.plot(
        model_pos[0], model_pos[1], 'ro')

    # Установка границ по осям X и Y чтобы видно было только рабочую область
    limits = model_view.working_area_limits(0.1)
    mv.set_xlim(*limits[0])
    mv.set_ylim(*limits[1])

    # ----------------------------------------

    start = time.time()

    # ----------------------------------------

    m = rezonator.get_metrics()

    mp = ax[2]

    # рисуем температуру
    tc, = mp.plot(dt.datetime.fromtimestamp(start), m['temperature'], 'r-')

    # рисуем изменение частоты
    fс, = mp.plot(dt.datetime.fromtimestamp(start), m['freq_change'], 'b-')

    # Рисуем диссбаланс
    dc, = mp.plot(dt.datetime.fromtimestamp(start), m['disbalance'], ':')

    # форматирование оси X
    # устанавливаем интервал в 1 секунду
    hours = mdates.SecondLocator(interval=1)
    time_format = mdates.DateFormatter('%S')  # устанавливаем формат времени
    mp.xaxis.set_major_locator(hours)  # устанавливаем локатор основных делений
    # устанавливаем форматтер основных делений
    mp.xaxis.set_major_formatter(time_format)

    # ----------------------------------------

    plt.show(block=False)

    cycle_time = 0

    while True:
        click = f.ginput(show_clicks=False, timeout=0.01)
        if len(click) != 0:
            click = click[0]

            current_pos.set_data(click[0], click[1])

            model_pos = playground.map_to_model(click)
            current_pos_transformed.set_data(model_pos[0], model_pos[1])

            match ModelView.detect_zone(model_pos):
                case Zone.BODY:
                    # just heat up
                    rezonator.heat_body(LASER_POWER, cycle_time)
                case Zone.FORBIDDEN:
                    # heat up and add energy to forbidden zone
                    rezonator.heat_forbidden(LASER_POWER, cycle_time)
                case Zone.TARGET1 | Zone.TARGET2 as zone:
                    # Обработка мишеней
                    pos = ModelView.map_to_zone(zone, model_pos)
                    rezonator.target(zone, pos, LASER_POWER, cycle_time)

                    # Обновление цвета мишеней
                    # Генерируем новую модель по результатам обновления
                    model_view = rezonator.get_model_view(offset, angle)
                    for i in range(2):
                        target_colors = model_view.target_color_map(i)
                        for row in zip(patches[i], target_colors):
                            for patch, color in zip(*row):
                                patch.set_facecolor(color)

        else:
            rezonator.idle(cycle_time)

        # ------------ Метрики -------------------

        m = rezonator.get_metrics()

        d = tc.get_data(orig=True)

        points = min(100, len(d[0]))
        ts = dt.datetime.fromtimestamp(time.time())

        d = [list(d[0][-points:]), list(d[1][-points:])]
        d[0].append(ts)  # type: ignore
        d[1].append(m['temperature'])
        tc.set_data(d)

        d = fс.get_data(orig=True)
        d = [list(d[0][-points:]), list(d[1][-points:])]
        d[0].append(ts)  # type: ignore
        d[1].append(m['freq_change'])
        fс.set_data(d)

        d = dc.get_data(orig=True)
        d = [list(d[0][-points:]), list(d[1][-points:])]
        d[0].append(ts)  # type: ignore
        d[1].append(m['disbalance'] * 100)
        dc.set_data(d)

        mp.relim()
        mp.autoscale_view()

        print(
            f"Static freq change: {m['static_freq_change']:.2f} Hz, disbalance: {m['disbalance'] * 100:.2f} %")

        # ----------------------------------------

        cycle_time = time.time() - start
        start = time.time()

        plt.draw()
