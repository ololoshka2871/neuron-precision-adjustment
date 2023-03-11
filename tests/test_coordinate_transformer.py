

import pytest

import numpy as np

from matplotlib.transforms import Affine2D

from misc.coordinate_transformer import CoordinateTransformer, RealCoordinates, ModelCoordinates, WorkzoneRelativeCoordinates
from misc.common import Rezonator


class TestCoordinateTransformer:
    @pytest.fixture
    def rezonator(self) -> Rezonator:
        return Rezonator.load()
    
    @pytest.fixture
    def transformer(self, rezonator) -> CoordinateTransformer:
        return CoordinateTransformer(rezonator, workzone_center=(10.0, 10.0))

    def test_wrap_from_real_to_model(self,
                                     rezonator: Rezonator,
                                     transformer: CoordinateTransformer):
        real = 10.0, 10.0
        model = 0.0, rezonator.work_zone_center_pos[1]
        model_coordinates = transformer.wrap_from_real_to_model(RealCoordinates(*real))
        assert model_coordinates == ModelCoordinates(*model)
        
    def test_wrap_from_real_to_model_back(self,
                                          transformer: CoordinateTransformer):
        real = RealCoordinates(10.0, 10.0)
        model_coordinates = transformer.wrap_from_real_to_model(real)
        real_coordinates = transformer.wrap_from_model_to_real(model_coordinates)
        assert real_coordinates == real
        
    def test_wrap_from_real_to_workzone_relative(self,
                                                 transformer: CoordinateTransformer):
        real = RealCoordinates(10.1, 10.1)
        wz_coordinates = transformer.wrap_from_real_to_workzone_relative(real)
        assert wz_coordinates == ModelCoordinates(0.0625, 0.13333333333333286)
        
    def test_wrap_from_real_to_workzone_relative_back(self,
                                                        transformer: CoordinateTransformer):
        real = RealCoordinates(10.0, 10.0)
        wz_coordinates = transformer.wrap_from_real_to_workzone_relative(real)
        real_coordinates = transformer.wrap_from_workzone_relative_to_real(wz_coordinates)
        assert real == real_coordinates
        
    def test_wrap_workzone_relative_to_model(self,
                                             rezonator: Rezonator,
                                             transformer: CoordinateTransformer):
        wz = WorkzoneRelativeCoordinates(0.0, 0.0)
        model = transformer.wrap_from_workzone_relative_to_model(wz)
        assert model == ModelCoordinates(0.0, rezonator.work_zone_center_pos[1])
        
    def test_chain_transform(self,
                             transformer: CoordinateTransformer):
        real = RealCoordinates(138.15, 6.24)
        model = transformer.wrap_from_real_to_model(real)
        wz = transformer.wrap_from_model_to_workzone_relative(model)
        rb = transformer.wrap_from_workzone_relative_to_real(wz)
        assert real == rb
        
    def test_chain_transform2(self,
                             transformer: CoordinateTransformer):
        wz = WorkzoneRelativeCoordinates(0, 1)
        model = transformer.wrap_from_workzone_relative_to_model(wz)
        real = transformer.wrap_from_model_to_real(model)
        wz_r = transformer.wrap_from_real_to_workzone_relative(real)
        assert wz == wz_r


class TestAffine2D:
    def test_inverted(self):
        t1 = Affine2D() \
            .rotate_deg_around(5, 3, 20) \
            .translate(10, 10)
        t2 = Affine2D() \
            .translate(-10, -10) \
            .rotate_deg_around(5, 3, -20)
            
        point = (5, 3)
        
        assert np.all(t2.transform_point(t1.transform_point(point)) == point)
        
    def test_order(self):
        t1 = Affine2D() \
            .rotate_deg_around(5, 3, 20) \
            .translate(10, 10)
        t2 = Affine2D() \
            .rotate_deg_around(5, 3, -20) \
            .translate(-10, -10) 
        
        point = (5, 3)
        
        assert np.all(t1.transform_point(point) == t1.transform_point(point))

# class TestCoordinateTransformerRandom:
#    @pytest.fixture
#    def transformer(self) -> CoordinateTransformer:
#        return CoordinateTransformer(Rezonator.load(),
#                                     offset=(np.random.random() * 0.3, np.random.random() * 0.5),
#                                     angle=np.random.random() * 20 - 10)
#
#    @pytest.mark.parametrize("test", [
#        ((0.0, 0.0)),
#        ((0.0, 1.0)),
#        ((1.0, 0.0)),
#        ((1.0, 1.0)),
#    ])
#    def test_wrap_from_real_to_model(self,
#                                     transformer: CoordinateTransformer,
#                                     test: tuple[float, float]):
#        t = RealCoordinates(test[0], test[1])
#        model_coordinates = transformer.wrap_from_real_to_model(t)
#        br_coordinates = transformer.wrap_from_model_to_real(model_coordinates)
#        assert br_coordinates == t
#
#    @pytest.mark.parametrize("test", [
#        ((0.0, 0.0)),
#        ((0.0, 1.0)),
#        ((1.0, 0.0)),
#        ((1.0, 1.0)),
#    ])
#    def test_wrap_from_real_to_workzone_relative(self,
#                                                 transformer: CoordinateTransformer,
#                                                 test: tuple[float, float]):
#        t = RealCoordinates(0.0, 0.0)
#        wz_coordinates = transformer.wrap_from_real_to_workzone_relative(t)
#        br_coordinates = transformer.wrap_from_workzone_relative_to_real(wz_coordinates)
#        assert br_coordinates == t
