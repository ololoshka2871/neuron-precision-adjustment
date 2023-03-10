

import pytest

import numpy as np

from misc.coordinate_transformer import CoordinateTransformer, RealCoordinates, ModelCoordinates
from misc.common import Rezonator


class TestMovement:
    @pytest.fixture
    def transformer(self) -> CoordinateTransformer:
        return CoordinateTransformer(Rezonator.load(), 
                                     offset=(np.random.random() * 0.3, np.random.random() * 0.5), 
                                     angle=np.random.random() * 20 - 10)

    @pytest.mark.parametrize("real, model", [
        (RealCoordinates(0.0, 0.0), ModelCoordinates(0.0, 0.0)),
        (RealCoordinates(1.0, 0.0), ModelCoordinates(1.0, 0.0)),
        (RealCoordinates(0.0, 1.0), ModelCoordinates(0.0, 1.0)),
        (RealCoordinates(1.0, 1.0), ModelCoordinates(1.0, 1.0)),
    ])
    def test_wrap_from_real_to_model(self, 
                                     transformer: CoordinateTransformer, 
                                     real: RealCoordinates, 
                                     model: ModelCoordinates):
        model_coordinates = transformer.wrap_from_real_to_model(real)
        assert model_coordinates == model
