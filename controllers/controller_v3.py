import numpy as np

import tensorflow as tf
from keras import Sequential, layers, optimizers
from keras.models import clone_model


class NNController:
    """
    Контроллер - нейронная сеть
    Входы:
        - freq_history - История измерений freq_history_size
        - move_history - История перемещений вида (side, step, S, F) длиной move_history_size
        - time - Относительное время до таймаута (0.0..1.0)

    Выходы:
        - Шаг по вертикали: -1.0..1.0
        - Желаемая скорость перемещения: 0.0..1.0
        - Желаемая выходная мощность лазера: 0.0..1.0
        - Самооценка: 0.0..1.0
    """

    INPUT_COUNT_CONST = 1
    OUTUT_COUNT = 2 + 1 + 1 + 1

    _model = None
    _history_len = 0
    _preend_neurons = 0
    _mean_layers = 0

    _total_inputs = 0

    @staticmethod
    def init_model(freq_history_size: int, move_history_size: int,
                   mean_layers=2, pre_end_layer_neurons=10) -> int:
        model = Sequential()

        NNController._total_inputs = freq_history_size + \
            (move_history_size * 4) + NNController.INPUT_COUNT_CONST

        # входы
        model.add(layers.Input(batch_size=1, shape=(NNController._total_inputs,)))
        # первый скрытый слой
        model.add(layers.Dense(units=NNController._total_inputs, activation='tanh'))

        # средние скрытые слои
        for _ in range(mean_layers - 1):
            model.add(layers.Dense(units=NNController._total_inputs, activation='elu'))

        # последний слой
        model.add(layers.Dense(units=pre_end_layer_neurons, activation='elu'))

        # выходной слой
        model.add(layers.Dense(units=NNController.OUTUT_COUNT, activation='tanh'))
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(0.1))
        model.trainable = False

        # model.summary()
        NNController._model = model

        NNController._history_size = freq_history_size
        NNController._mean_layers = mean_layers
        NNController._preend_neurons = pre_end_layer_neurons

        weights = model.get_weights()
        flat_weights = NNController._convert_weights_from_model(weights)
        return len(flat_weights)  # type: ignore

    @staticmethod
    def _convert_weights_to_model(weigths: list[float]):
        # должен получиться список следующего вида
        # Слой 0 - все веса от нейронов входа
        # Слой 0 - все веса смещенией
        # Слой 1...
        ws = NNController._model.get_weights()  # type: ignore
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

    @staticmethod
    def shuffled_weights() -> list[float]:
        """
        Возвращает случайно сгенерированные веса нейронной сети
        """
        weights = NNController._model.get_weights()  # type: ignore
        all_weights = []
        for l in weights:
            orig_shape = l.shape
            if len(orig_shape) > 1:
                rs = l.reshape(orig_shape[0] * orig_shape[1],)
                np.random.shuffle(rs)
                all_weights.extend(rs)
            else:
                all_weights.extend(l)
        return all_weights

    def __init__(self, wieghts: list | None = None, save_history=False):
        """
        Создет контроллер с указанными весами нейронной сети
        """
        self._model = clone_model(model=NNController._model)
        if wieghts is not None:
            self._model.set_weights(
                NNController._convert_weights_to_model(wieghts))
        self._model.trainable = False
        if save_history:
            self._input_history = np.empty(shape=(1, NNController._total_inputs), dtype=np.float32)
        else:
            self._input_history = None

    def update(self, input: dict):
        """
        Функция обновляет состояние контроллера
        """

        v = list()
        
        v.extend(input['freq_history'].flatten())
        v.extend(input['move_history'].flatten())
        v.append(input['time'])

        v = [v]

        if self._input_history is not None:
            self._input_history = np.append(self._input_history, v, axis=0)

        input = tf.convert_to_tensor(v, dtype=tf.float32)  # type: ignore
        output, = self._model(input)  # type: ignore
        output = output.numpy()

        speed = NNController.map_zero_one(output[2])
        return {
            'step': output[0],
            'power': NNController.map_zero_one(output[1]),
            'speed': speed if speed > 0 else 0.01,
            'self_grade': NNController.map_zero_one(output[3])
        }

    def get_weights(self):
        weights = self._model.get_weights()  # type: ignore
        return NNController._convert_weights_from_model(weights)

    def history(self) -> np.ndarray:
        if self._input_history is None:
            return np.array([])
        else:
            return self._input_history