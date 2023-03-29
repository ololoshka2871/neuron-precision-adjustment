import math

from matplotlib.transforms import Affine2D
import numpy as np

from typing import Optional

import gymnasium as gym
from gymnasium import spaces

import pygame

from misc.Rezonator import Rezonator
from misc.coordinate_transformer import CoordinateTransformer, RealCoordinates, WorkzoneRelativeCoordinates
from misc.f_s_transformer import FSTransformer
from models.movement import Movment

from models.rezonator_model import ModelView, RezonatorModel, Zone


class QuartzEnv4(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self,
                 render_mode=None,
                 time_limit: float = math.inf,
                 rezonator_thickness: float = 0.23,
                 heat_dissipation_rate: float = 0.9,
                 ambient_temperatire: float = 0.0,
                 power_threshold: float = 0.05,
                 layer_thikness_mean=0.5e-3,
                 tfk: float = -0.05,
                 adjust_mean: float = 10.0,
                 modeling_period: float = 0.01,
                 freqmeter_period: float = 0.4,
                 laser_power: float = 30.0,
                 wait_multiplier: float = 1.0,
                 relative_move = False):
        self.window_size = 1024  # The size of the PyGame window

        """
        Регистрация измеряемых параметров. То, что будет отдавать модель на каждом шаге
        - (0, 1): laser_position - Текущее положение лазера
        - 2: frequency - Текущая частота
        - 3: freq_change - Изменение частоты по сравнению с изначальной
        """
        self.observation_space = spaces.Box(
            np.array([-1.0, -1.0, 0.0, 0.0, -math.inf, 0.0],
                     dtype=np.float32),  # type: ignore
            np.array([1.0, 1.0, 1.0, math.inf, math.inf, math.inf],
                     dtype=np.float32),  # type: ignore
            dtype=np.float32)

        """
        Возможные действия
        - 0: Признак действия:
            - <1: move - Передвинуть лазер на заданное расстояние
            - <2: set_power - Изменить мощность лазера
            - <3: wait - Ожидаение
            - <4: end - Закончить эпизод
        - 1: x - Координата x, если move, иначе power или время ожидания
        - 2: y - Координата y
        - 3: F - Скорость перемещения
        """
        self.action_space = spaces.Box(
            np.array([0.0, -1.0, -1.0, 0.0], dtype=np.float32),  # type: ignore
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),  # type: ignore
            dtype=np.float32)
        self.action_count = 4

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self._rez = Rezonator.load()

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.rezonator_thickness = rezonator_thickness
        self.heat_dissipation_rate = heat_dissipation_rate
        self.ambient_temperatire = ambient_temperatire
        self.power_threshold = power_threshold
        self.layer_thikness_mean = layer_thikness_mean
        self.tfk = tfk
        self.adjust_mean = adjust_mean
        self._modeling_period = modeling_period
        self._freqmeter_period = freqmeter_period
        self._laser_power = laser_power
        self._wait_multiplier = wait_multiplier
        self._time_limit = time_limit
        self._relative_move = relative_move

        self._movement = Movment()
        self._rezonator_model: Optional[RezonatorModel] = None
        self._coord_transformer = None
        self._current_position = WorkzoneRelativeCoordinates(0.0, 1.0)
        self._prev_position = WorkzoneRelativeCoordinates(0.0, 1.0)
        self._params = {}
        self._f_s_transformer = None
        self._prev_freq = None
        self._transform = None
        self._lastact = None

        self._step_counter = 0
        self._current_power = 0.0
        self._current_speed = 0.0
        self._next_mesure_after = 0.0

    def _get_obs(self):
        """
        Возвращает текущее наблюдаемое состояние среды
        """
        if self._rezonator_model is None:
            raise RuntimeError(
                "Rezonator is not initialized, call reset() first!")

        rm = self._rezonator_model.get_metrics()
        return np.array([
            *self._current_position,
            self._current_power,
            rm['freq_change'],
            self._params['adjust_target'],
            self._time_elapsed], dtype=np.float32)

    def _get_info(self):
        """
        Возвращает полную информацию о среде
        """
        assert self._rezonator_model is not None

        rm = self._rezonator_model.get_metrics()
        return {
            # "params": self._params,
            "current_power": self._current_power,
            "current_speed": self._current_speed,
            "static_freq_change": rm['static_freq_change'],
            "temperature": rm['temperature'],
            "disbalance": rm['disbalance'],
            "penalty_energy": rm['penalty_energy'],
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        self._params['offset'] = (
            options is not None and 'offset' in options) or (np.random.random() * 0.3, np.random.random() * 0.5)
        self._params['angle'] = (
            options is not None and 'angle' in options) or np.random.random() * 20 - 10
        self._params['layer_thikness'] = (
            options is not None and 'layer_thikness' in options) or np.clip(self.np_random.normal(
                self.layer_thikness_mean, 0.1e-3), a_min=0.4e-3, a_max=0.6e-3)
        self._params['adjust_target'] = (
            options is not None and 'adjust_target' in options) or np.clip(
                self.np_random.normal(
                    self.adjust_mean, self.adjust_mean * 0.1),
                a_min=self.adjust_mean * 0.5, a_max=self.adjust_mean * 1.5)

        max_s = options is not None and 'max_s' in options or 255.0
        max_f = options is not None and 'max_f' in options or 1000.0

        self._f_s_transformer = FSTransformer(max_s, max_f)
        self._current_position = WorkzoneRelativeCoordinates(0.0, 1.0)
        self._prev_position = WorkzoneRelativeCoordinates(0.0, 1.0)

        self._rezonator_model = RezonatorModel(
            rezonator_thickness=self.rezonator_thickness,
            heat_dissipation_rate=self.heat_dissipation_rate,
            ambient_temperatire=self.ambient_temperatire,
            power_threshold=self.power_threshold,
            layer_thikness=self._params['layer_thikness'],
            tfk=self.tfk
        )

        self._coord_transformer = CoordinateTransformer(
            resonator=self._rez,
            workzone_center=(0, 0),
            offset=self._params['offset'],
            angle=self._params['angle']
        )
        self._step_counter = 0
        self._current_power = 0.0
        self._current_speed = 0.0
        self._next_mesure_after = 0.0
        self._time_elapsed = 0.0

        self._lastact = None
        self._transform = None

        self._prev_freq = self._rezonator_model.get_metrics()['freq_change']

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _decode_action(self, action: np.ndarray) -> dict:
        assert action[0] >= 0.0 and action[0] <= 1.0
        action_code = action[0] * self.action_count

        def rearrange(x: float):
            return (x + 1.0) / 2.0

        if action_code < 1:
            return {'Action': 'Move', 'X': action[1], 'Y': action[2], 'F': rearrange(action[3])}
        elif action_code < 2:
            return {'Action': 'SetPower', 'Power': rearrange(action[1])}
        elif action_code < 3:
            return {'Action': 'Wait', 'Time': rearrange(action[1]) * self._wait_multiplier}
        elif action_code <= 4:
            return {'Action': 'End'}
        else:
            raise RuntimeError("Invalid action code")

    def step(self, action: np.ndarray):
        self._lastact = self._decode_action(action)

        terminated = False
        reward = 0.0

        match self._lastact['Action']:
            case 'Move':
                reward = self._sim_step(
                    self._lastact['X'], self._lastact['Y'], self._lastact['F'])
            case 'SetPower':
                self._current_power = self._lastact['Power']
                reward = -0.5
                self._time_elapsed += 0.1
            case 'Wait':
                reward = self._wait_on(self._lastact['Time'])
            case 'End':
                reward = self._finalise()
                terminated = True

        self._lastact['rev'] = reward
        self._step_counter += 1

        # --------------------

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        trancated = self._time_elapsed >= self._time_limit

        return observation, reward, terminated, trancated, info

    def render(self):
        """
        Нарисовать текущее состояние среды
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        """
        Закрыть окно и освободить ресурсы
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _render_frame(self):
        """
        Рисует текущее состояние среды в модельных координатах.
        """

        assert self._coord_transformer is not None
        assert self._rezonator_model is not None

        # базовая точка - середина в месте крепления (0, 0)
        rezonator = self._rez['rezonator']

        # рабочая область
        working_area = self._rez['working_area']

        # Запрещенная область
        forbidden_area = self._rez['forbidden_area']

        model_view = self._rezonator_model.get_model_view(
            self._params['offset'], self._params['angle'])

        real_rezonator = self._coord_transformer.array_wrap_from_model_to_real(
            rezonator)
        real_forbidden_area = self._coord_transformer.array_wrap_from_model_to_real(
            forbidden_area)
        real_working_area = self._coord_transformer.get_real_working_zone(
            working_area)

        model_rezonator = self._coord_transformer.array_wrap_from_real_to_model(
            real_rezonator)
        model_forbidden_area = self._coord_transformer.array_wrap_from_real_to_model(
            real_forbidden_area)
        model_working_area = self._coord_transformer.array_wrap_from_real_to_model(
            real_working_area)

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self._transform is None:
            model_working_area_x_min = np.min(model_working_area[:, 0])
            model_working_area_x_max = np.max(model_working_area[:, 0])
            model_working_area_y_min = np.min(model_working_area[:, 1])
            model_working_area_y_max = np.max(model_working_area[:, 1])
            model_working_area_x_size = model_working_area_x_max - model_working_area_x_min
            model_working_area_y_size = model_working_area_y_max - model_working_area_y_min

            self._transform = Affine2D() \
                .translate(-model_working_area_x_min, -model_working_area_y_min) \
                .scale(self.window_size / model_working_area_x_size, self.window_size / model_working_area_y_size)

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pygame.Surface.lock(canvas)
        # Рисуем резонатор
        pc_rezonator = self._transform.transform(  # type: ignore
            model_rezonator)
        pygame.draw.polygon(canvas, points=pc_rezonator,  # type: ignore
                            width=1, color='black')
        # Рисуем Запрещенную область
        pc_forbidden_area = self._transform.transform(  # type: ignore
            model_forbidden_area)
        pygame.draw.polygon(canvas, points=pc_forbidden_area,  # type: ignore
                            width=0, color='magenta')
        # Рисуем рабочую область
        pc_working_area = self._transform.transform(  # type: ignore
            model_working_area)
        pygame.draw.polygon(canvas, points=pc_working_area,  # type: ignore
                            width=1, color='blue')
        # Рисуем цели
        for i in range(2):
            for row in zip(model_view.target(i), model_view.target_color_map(i)):
                for rect, color in zip(*row):
                    xy = rect.get_xy()
                    wh = (rect.get_width(), rect.get_height())
                    xy2 = (xy[0] + wh[0], xy[1] + wh[1])
                    xy_t = self._transform.transform(xy)  # type: ignore
                    xy2t = self._transform.transform(xy2)  # type: ignore
                    rect = pygame.Rect(
                        xy_t,
                        xy2t - xy_t
                    )
                    color = pygame.color.Color(
                        int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
                    pygame.draw.rect(canvas, rect=rect, width=0,
                                     color=color)  # type: ignore

        # Рисуем траекторию
        color = pygame.color.Color(
            int(self._current_speed * 255), 0, 0, int(self._current_power * 255))
        src = self._transform.transform(  # type: ignore
            self._coord_transformer.wrap_from_workzone_relative_to_model(self._prev_position).tuple())
        dest = self._transform.transform(  # type: ignore
            self._coord_transformer.wrap_from_workzone_relative_to_model(
                self._current_position).tuple())
        pygame.draw.circle(canvas, center=src, radius=5,  # type: ignore
                           width=0, color=color)
        pygame.draw.circle(canvas, center=dest, radius=5,  # type: ignore
                           width=0, color=color)
        pygame.draw.line(canvas, start_pos=src, end_pos=dest,  # type: ignore
                         width=2, color=color)

        # pygame.draw.line(canvas, start_pos=self._transform.transform(  # type: ignore
        #    (0, -5)), end_pos=self._transform.transform((0, 5)), width=1, color='black')  # type: ignore
        # pygame.draw.line(canvas, start_pos=self._transform.transform(  # type: ignore
        #    (-5, 0)), end_pos=self._transform.transform((5, 0)), width=1, color='black')  # type: ignore

        pygame.Surface.unlock(canvas)

        canvas = pygame.transform.flip(canvas, False, True)

        x, y = self._current_position
        font = pygame.font.SysFont('Arial', 20)
        text = font.render(
            f"Current: x: {x:.2f}, y: {y:.2f}, S: {self._current_power:.2f}, F:{self._current_speed:.2f}, T: {self._time_elapsed:.2f} s.",
            True, (0, 0, 0, 0))
        canvas.blit(text, (10, 10))
        text = font.render(f"Act: {self._lastact_str()}", True, (0, 0, 0, 0))
        canvas.blit(text, (10, 35))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())  # type: ignore
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])  # type: ignore
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _sim_step(self, x: float, y: float, F: float) -> float:
        """
        Выполняет один шаг симуляции
        :param x: координата x
        :param y: координата y
        :param F: Скорость перемещения
        :return: награда за шаг
        """

        assert self._rezonator_model is not None
        assert self._f_s_transformer is not None
        assert self._coord_transformer is not None
        assert self._prev_freq is not None

        if self._relative_move:
            dest_wz, clipped = WorkzoneRelativeCoordinates(x, y).add(
                self._current_position).clip(-1.0, 1.0, -1.0, 1.0)  # absolute in work zone
        else:
            dest_wz, clipped = WorkzoneRelativeCoordinates(x, y).clip(
                -1.0, 1.0, -1.0, 1.0)
        dest_real = self._coord_transformer.wrap_from_workzone_relative_to_real(
            dest_wz)
        src_real = self._coord_transformer.wrap_from_workzone_relative_to_real(
            self._current_position)

        self._current_speed = np.clip(F, 1e-3, 1.0)  # speed mast be > 0
        traectory = self._movement.interpolate_move(
            src=src_real.tuple(),
            dst=dest_real.tuple(),
            speed=self._f_s_transformer.map_f_to_global(self._current_speed),
            time_step=self._modeling_period,
        )

        total_reward = dest_real.abs_path_from(src_real) * 0.1 if not clipped else -1.0
        last_zone = Zone.NONE
        for pos_x, pos_y, _ in zip(*traectory):
            self._next_mesure_after -= self._modeling_period
            if self._next_mesure_after <= 0.0:
                self._next_mesure_after = self._freqmeter_period

                m = self._rezonator_model.get_metrics()
                freq_change = self._prev_freq - m['freq_change']
                if freq_change < 0.0:
                    total_reward += 1.0

            model_pos = self._coord_transformer.wrap_from_real_to_model(
                RealCoordinates(pos_x, pos_y)).tuple()

            laser_power = self._laser_power * self._current_power  # мощность лазера в Вт

            zone = ModelView.detect_zone(model_pos, last_zone)
            match zone:
                case Zone.BODY:
                    # just heat up
                    self._rezonator_model.heat_body(
                        laser_power, self._modeling_period)
                case Zone.FORBIDDEN:
                    # heat up and add energy to forbidden zone
                    self._rezonator_model.heat_forbidden(
                        laser_power, self._modeling_period)
                case Zone.TARGET1 | Zone.TARGET2 as zone:
                    # Обработка мишеней
                    pos = ModelView.map_to_zone(zone, model_pos)
                    self._rezonator_model.target(
                        zone, pos, laser_power, self._modeling_period)
                case _:
                    self._rezonator_model.idle(self._modeling_period)
            last_zone = zone

        self._prev_position = self._current_position
        self._current_position = dest_wz  # обновляем текущую позицию
        self._time_elapsed += traectory[2][-1]

        return total_reward

    def _wait_on(self, wait_time: float) -> float:
        """
        Стоит на месте в течение `wait_time` секунд
        """

        assert self._coord_transformer is not None
        assert self._rezonator_model is not None
        assert self._prev_freq is not None

        model_pos = self._coord_transformer.wrap_from_workzone_relative_to_model(
            self._current_position).tuple()
        laser_power = self._laser_power * self._current_power  # мощность лазера в Вт
        zone = ModelView.detect_zone(model_pos)

        match zone:
            case Zone.BODY:
                # just heat up
                def f(t): return self._rezonator_model.heat_body(  # type: ignore
                    laser_power, t)
            case Zone.FORBIDDEN:
                # heat up and add energy to forbidden zone
                def f(t): return self._rezonator_model.heat_forbidden(  # type: ignore
                    laser_power, t)
            case Zone.TARGET1 | Zone.TARGET2 as zone:
                # Обработка мишеней
                pos = ModelView.map_to_zone(zone, model_pos)
                def f(t): return self._rezonator_model.target(  # type: ignore
                    zone, pos, laser_power, t)
            case _:
                def f(t): return self._rezonator_model.idle(t)  # type: ignore

        self._time_elapsed += wait_time

        if wait_time < self._next_mesure_after:
            f(wait_time)
            self._next_mesure_after -= wait_time
            return 0.0
        else:
            total_reward = 0.0
            while wait_time > self._next_mesure_after:
                f(self._next_mesure_after)
                wait_time -= self._next_mesure_after
                self._next_mesure_after = self._freqmeter_period

                m = self._rezonator_model.get_metrics()
                freq_change = self._prev_freq - m['freq_change']
                self._prev_freq = m['freq_change']
                if freq_change < 0.0:
                    total_reward += -freq_change

            f(wait_time)
            self._next_mesure_after -= wait_time
            return total_reward

    def _finalise(self) -> float:
        """
        Вызывается после окончания эпизода, подстчитывает штрафы
        """

        assert self._rezonator_model is not None
        assert self._coord_transformer is not None

        # Штраф за то, что лазер не выведен за пределы резонатора
        model_pos = self._coord_transformer.wrap_from_workzone_relative_to_model(
            self._current_position).tuple()
        zone = ModelView.detect_zone(model_pos)
        match zone:
            case Zone.NONE:
                zone_penalty = 0.0
            case _:
                zone_penalty = 1.0

        rezonator_metrics = self._rezonator_model.get_metrics()

        # Сколько частоты изменилось за эпизод
        freq_change = rezonator_metrics['static_freq_change']
        db = abs(rezonator_metrics['disbalance'])
        penalty = rezonator_metrics['penalty_energy']
        # Сколько надо было настроить
        adjust_target = self._params['adjust_target']

        # Отклонение от цели
        # <0 - перебор
        # 0 - идеально
        # >0 - недобор
        freq_target_distance_rel = (
            adjust_target - freq_change) / adjust_target

        adjust_penalty = freq_target_distance_rel if freq_target_distance_rel > 0.0 else - \
            freq_target_distance_rel * 2.0

        wieghts = np.array([-10.0, -5.0, -100.0, -50.0])
        values = np.array([adjust_penalty, db, penalty, zone_penalty])

        res = values * wieghts

        return res.sum()

    def _lastact_str(self) -> str:
        """
        Возвращает строку с последним действием
        """

        if self._lastact is None:
            return '0'
        else:
            match self._lastact['Action']:
                case 'Move':
                    return f"{self._step_counter} Move step: X{self._lastact['X']:.2f} Y{self._lastact['Y']:.2f} F{self._lastact['F']:.2f}, rev={self._lastact['rev']}"
                case 'SetPower':
                    return f"{self._step_counter} SetPower: {self._lastact['Power']:.2f}, rev={self._lastact['rev']}"
                case 'Wait':
                    return f"{self._step_counter} Wait: {self._lastact['Time']:.2f} s., rev={self._lastact['rev']}"
                case _:
                    return 'UNKNOWN'
