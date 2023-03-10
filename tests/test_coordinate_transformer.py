

import pytest

import numpy as np

from misc.coordinate_transformer import CoordinateTransformer, RealCoordinates, ModelCoordinates
from misc.common import Rezonator


class TestMovement:
    @pytest.fixture
    def transformer(self):
        return CoordinateTransformer(Rezonator.load(), offset=(0.0, 0.0), angle=0.0)

    @pytest.mark.parametrize("real, model", [
        (RealCoordinates(0.0, 0.0), ModelCoordinates(0.0, 0.0)),
    ])
    def test_wrap_from_real_to_model(self, transformer, real, model):
        model_coordinates = transformer.wrap_from_real_to_model(real)
        assert model_coordinates == model
