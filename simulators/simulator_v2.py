import numpy as np

from misc.queue import Queue
from models.rezonator_model import RezonatorModel, Zone, ModelView
from models.movement import Movment
from misc.coordinate_transformer import CoordinateTransformer, WorkzoneRelativeCoordinates, RealCoordinates
from misc.f_s_transformer import FSTransformer
from models.sim_stop_detector_v2 import SimStopDetector
from models.stop_condition import StopCondition


class Simulator:
    """
    Симуляция обработки лазером резонатора v2
    Отличие от v1:
    - Вместо расчета положения станка по точкам можно сразу расчитать путь с указанным шагом и скоростью
    - Если текущее заданное перемещение оказывается дольше, чем времение до следующего измерения, 
        то расчитывается расчитывается кусок до изверения, вычислятся новое измерение и продолжается движение до конца 
        или следующего измерения. Таким образом избагается множественный вызов расчета нейронной сети по 1 шагу
    - Нейронной сети на вход передаются не просто координаты текущего положения станка, а история последних move_history_len
        положений в виде наборов {dest_ps, S, F}.
    - Благодоря тому факту, что ели движение запущено его нельзя прервать или изменить цель то просто соберем историю измнений
        частоты и не будем считать нейронку каждый шаг симуляции, а будем считать только по окончанию движения, чтобы получить 
        новую команду на движение
    """

    def __init__(self,
                 rezonator_model: RezonatorModel,
                 controller_v2,
                 coord_transformer: CoordinateTransformer,
                 fs_transformer: FSTransformer,
                 laser_power: float,
                 initial_freq_diff=0.5,
                 freqmeter_period=0.4,
                 modeling_period=0.01,
                 freq_history_size=10,
                 move_history_len=10,
                 initial_wz_pos=WorkzoneRelativeCoordinates(0.0, 1.0)):
        """
        :param rezonator_model: Модель резонатора
        :param controller_v2: Контроллер
        :param coord_transformer: Преобразователь координат
        :param fs_transformer: Преобразователь мощности лазера и скорости перемещения станка
        :param laser_power: Мощность лазера в Вт
        :param initial_freq_diff: Относительное начальное отклонение частоты [0..1]
        :param freqmeter_period: Период измерения частоты
        :param modeling_period: Период моделирования
        :param freq_history_size: Размер истории измерений частоты
        :param move_history_len: Размер истории перемещений
        :param initial_wz_pos: Начальное положение станка в рабочей зоне
        """
        self._rezonator_model = rezonator_model
        self._controller = controller_v2
        self._freqmeter_period = freqmeter_period
        self._modeling_period = modeling_period
        self._laser_power = laser_power

        self._coord_transformer = coord_transformer
        self._fs_transformer = fs_transformer

        self._period_accum = 0.0
        self._next_mesure_after = 0.0

        self._movement = Movment()
        self._curremt_pos_global = self._coord_transformer.wrap_from_workzone_relative_to_real(
            initial_wz_pos)

        self._initial_freq_diff = initial_freq_diff

        self._measure_diff_history = Queue(shape=(freq_history_size, 1))
        self._measure_diff_history.enqueue(
            np.repeat(initial_freq_diff, freq_history_size))

        self._move_history = Queue(shape=(move_history_len, 4))

        # (dest_pos_x, dest_pos_y, S, F)
        self._move_history.enqueue(
            np.reshape(
                np.repeat(
                    np.array([0.0, 1.0, 0.0, 0.0]), move_history_len),
                newshape=(4, move_history_len)).T)

    def perform_modeling(self,
                         stop_detector: SimStopDetector,
                         input_display=lambda input: None
                         ) -> StopCondition:
        """
        Выполнение симуляции
        :param stop_detector: Детектор условия остановки симуляции
        :param input_display: Колбэк-функция отображения входных данных нейронной сети
        :return: Причина остановки симуляции
        """

        while True:
            controller_input = dict(
                time=self._period_accum,
                freq_history=self._measure_diff_history.peek_all(),
                move_history=self._move_history.peek_all(),
                energy=stop_detector.get_energy_relative(),
            )
            input_display(controller_input)
            command = self._controller.update(controller_input)

            dest_wz = WorkzoneRelativeCoordinates(*command['destination'])
            dest_real = \
                self._coord_transformer.wrap_from_workzone_relative_to_real(
                    dest_wz)
            traectory = self._movement.interpolat_move_time_limit(
                src=self._curremt_pos_global.tuple(),
                dst=dest_real.tuple(),
                speed=self._fs_transformer.map_f_to_global(max(1e-3, command['speed'])), # speedm mast be > 0
                time_step=self._modeling_period,
                time_limit=stop_detector.get_time_limit() - self._period_accum,
            )

            self._curremt_pos_global = dest_real
            cmd_s = command['power']

            ts = 0.0  # заглушка
            last_zone = Zone.NONE
            for pos_x, pos_y, ts in zip(*traectory):
                self._next_mesure_after -= self._modeling_period
                if self._next_mesure_after <= 0.0:
                    self._next_mesure_after = self._freqmeter_period

                    # Сдвигаем историю измерений частоты и добавляем новое измерение
                    self._measure_diff_history.dequeue()
                    m = self._rezonator_model.get_metrics()
                    abs_freq_change = np.array(
                        [self._initial_freq_diff - self._map_adj_to_relatie(m['freq_change'])])
                    self._measure_diff_history.enqueue(abs_freq_change)

                model_pos = self._coord_transformer.wrap_from_real_to_model(
                    RealCoordinates(pos_x, pos_y)).tuple()

                laser_power = self._laser_power * cmd_s  # мощность лазера в Вт

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

            self._period_accum += ts

            prew_wz = self._move_history.peek_last_N(1)[0]
            self._shift_move_history(
                dest_wz.tuple(), S=cmd_s, F=command['speed'])

            last_freq = self._measure_diff_history.peek_rear()
            reason = stop_detector(
                self._period_accum, self._rezonator_model.get_metrics(),
                {
                    'F': command['speed'],
                    'S': command['power'],
                    'self_grade': command['self_grade'],
                    'Passed': dest_wz.abs_path_from(WorkzoneRelativeCoordinates(prew_wz[0], prew_wz[1])),
                }, self._rezonator_model.get_metrics()['freq_change'])
            if reason != StopCondition.NONE:
                return reason

    def _shift_move_history(self, dest_pos: tuple[float, float], S: float, F: float):
        self._move_history.dequeue()
        self._move_history.enqueue(np.array([[*dest_pos, S, F]]))

    def _map_adj_to_relatie(self, adj: float) -> float:
        """
        Преобразование изменения частоты резонатора в относительное значение приведенное к максимальному теоритическому изменинию
        :param adj: Абсолютное значение изменения частоты
        :return: Относительное значение
        """
        return adj / self._rezonator_model.possible_freq_adjust

    def use_rezonator(self, f, *args, **kwargs):
        """
        Применить функцию к резонатору, сохраняя его состояние
        """
        return f(self._rezonator_model, *args, **kwargs)
