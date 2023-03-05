import numpy as np

import tensorflow as tf
from keras import Sequential, layers, optimizers
from keras.models import clone_model


class RandomController:
    """
    Заглушка для контроллера, вмето вызодных параметров возвращает случайные значения в заданных диопазонах
    - Цель движения: tuple[x, y]: -1.0..1.0
    - Скорость перемещения: 0.0..1.0
    - Выходная мощность лазера: 0.0..1.0
    - Самооценка: -1.0..1.0
    """

    def __init__(self) -> None:
        pass

    def update(self, input: dict):
        """
        Функция обновляет состояние контроллера
        """

        def limitmp(x): return RandomController._limit(x, -1.0, 1.0)
        def limitzp(x): return RandomController._limit(x, 0.0, 1.0)

        return {
            'destination': list(map(limitmp, np.random.normal(0, 1.0, 2))),
            'power': limitzp(np.random.normal(0.5, 0.5)),
            'speed': RandomController._limit(np.random.normal(0.5, 0.5), 0.05, 1.0),
            'self_evaluation': limitmp(np.random.normal(0, 1.0))
        }

    @staticmethod
    def _limit(v: float, _min: float, _max: float) -> float:
        return max(_min, min(v, _max))


class NNController:
    """
    Контроллер - нейронная сеть
    Входы:
        - freq_history - История измерений
        - current_pos - относительная текущая позиция 
        - current_s - текущая мощность лазера
        - current_f - текущая скорость

    Выходы:
        - Цель движения: tuple[x, y]: -1.0..1.0
        - Скорость перемещения: 0.0..1.0
        - Выходная мощность лазера: 0.0..1.0
        - Самооценка: -1.0..1.0
    """

    INPUT_COUNT = 4
    OUTUT_COUNT = 5

    _model = None
    _history_len = 0
    _preend_neurons = 0
    _mean_layers = 0

    @staticmethod
    def init_model(history_size: int, mean_layers=2, preend_layer_neurons=10) -> int:
        model = Sequential()

        # входы
        model.add(layers.Input(batch_size=1, shape=(
            history_size + NNController.INPUT_COUNT,)))
        # первый скрытый слой
        model.add(layers.Dense(units=history_size +
                  NNController.INPUT_COUNT, activation='tanh'))

        # средние скрытые слои
        for _ in range(mean_layers - 1):
            model.add(layers.Dense(units=history_size +
                                   NNController.INPUT_COUNT, activation='elu'))

        # последний слой
        model.add(layers.Dense(units=preend_layer_neurons, activation='elu'))

        # выходной слой
        model.add(layers.Dense(units=NNController.OUTUT_COUNT, activation='tanh'))
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(0.1))
        
        #model.summary()
        NNController._model = model

        NNController._history_size = history_size
        NNController._mean_layers = mean_layers
        NNController._preend_neurons = preend_layer_neurons

        weights = model.get_weights()
        flat_weights = NNController._convert_weights_from_model(weights)
        return len(flat_weights)  # type: ignore

    @staticmethod
    def _convert_weights_to_model(weigths: list[float]):
        # должен получиться список следующего вида
        # Слой 0 - все веса от нейронов входа
        # Слой 0 - все веса смещенией
        # Слой 1...
        ws = NNController._model.get_weights() # type: ignore
        rp = 0
        for ln in range(len(ws)):
            orig_shape = ws[ln].shape
            if len(orig_shape) > 1:
                sz = orig_shape[0] * orig_shape[1]
                nd = np.reshape(weigths[rp:rp + sz], newshape=orig_shape)
            else:
                sz = orig_shape[0]
                nd = np.array(weigths[rp:rp + sz])
            ws[ln] = nd
            rp += sz
        return ws

    @staticmethod
    def _convert_weights_from_model(weigths) -> list[float]:
        all_weights = []
        for l in weigths:
            orig_shape = l.shape
            if len(orig_shape) > 1:
                rs = l.reshape(orig_shape[0] * orig_shape[1],)
                all_weights.extend(rs)
            else:
                all_weights.extend(l)
        return all_weights

    @staticmethod
    def map_zero_one(v: float) -> float:
        return (1.0 + v) / 2.0

    def __init__(self, wieghts: list | None = None):
        """
        Создет контроллер с указанными весами нейронной сети
        """
        self._model = clone_model(model=NNController._model)
        if wieghts is not None:
            self._model.set_weights(
                NNController._convert_weights_to_model(wieghts))

    def update(self, input: dict):
        """
        Функция обновляет состояние контроллера
        """

        v = [*input['freq_history'], *input['current_pos'],
             input['current_s'], input['current_f']]
        input = tf.constant([v])  # type: ignore
        output, = self._model.predict(input, verbose=None)  # type: ignore

        return {
            'destination': output[:2],
            'power': NNController.map_zero_one(output[2]),
            'speed': NNController.map_zero_one(output[3]),
            'self_grade': NNController.map_zero_one(output[4])
        }

    def get_weights(self):
        weights = self._model.get_weights()  # type: ignore
        return NNController._convert_weights_from_model(weights)
