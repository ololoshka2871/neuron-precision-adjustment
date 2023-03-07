#!/usr/bin/env python

import numpy as np

from misc.common import transform_all

# Константы
REZONATOR_LENGTH = 4.5
REZONATOR_HEIGHT = 1.0
BRANCH_LENGTH = 1.96
BRANCH_WIDTH = 0.38
REZONATOR_HEIGHT_HALF = REZONATOR_HEIGHT / 2

TARGET_WIDTH = 0.5
TARGET_HEIGHT = 0.32
TARGET_VERTICAL_OFFSET = 0.13
TARGETS_STEP = 0.62

TARGET_RIGHT_OFFSET = REZONATOR_HEIGHT_HALF - \
    BRANCH_WIDTH + (BRANCH_WIDTH - TARGET_HEIGHT) / 2

WORKING_AREA_Y_OFFSET = 3.67
WORKING_AREA_WIDTH = 3.2
WORKING_AREA_HEIGHT = 1.5
WORKING_AREA_WIDTH_HALF = WORKING_AREA_WIDTH / 2

FORBIDDEN_AREA_DOWN_OFFSET = 0.1
FORBIDDEN_AREA_SIDE_OFFSET = 0.01
FORBIDDEN_AREA_UP_OFFSET = 0.725


def rezonator_as_object():
    # Резонатор
    rezonator = np.array([[0, 0], [REZONATOR_HEIGHT_HALF, 0],
                        [REZONATOR_HEIGHT_HALF, REZONATOR_LENGTH],
                        [REZONATOR_HEIGHT_HALF - BRANCH_WIDTH, REZONATOR_LENGTH],
                        [REZONATOR_HEIGHT_HALF - BRANCH_WIDTH,
                            REZONATOR_LENGTH - BRANCH_LENGTH],
                        [-REZONATOR_HEIGHT_HALF + BRANCH_WIDTH,
                            REZONATOR_LENGTH-BRANCH_LENGTH],
                        [-REZONATOR_HEIGHT_HALF + BRANCH_WIDTH, REZONATOR_LENGTH],
                        [-REZONATOR_HEIGHT_HALF, REZONATOR_LENGTH],
                        [-REZONATOR_HEIGHT_HALF, 0],
                        [0, 0]])

    # первая ветка
    target1 = np.array([[TARGET_RIGHT_OFFSET, REZONATOR_LENGTH - TARGET_VERTICAL_OFFSET - TARGET_WIDTH],
                        [TARGET_RIGHT_OFFSET + TARGET_HEIGHT, REZONATOR_LENGTH -
                            TARGET_VERTICAL_OFFSET - TARGET_WIDTH],
                        [TARGET_RIGHT_OFFSET + TARGET_HEIGHT,
                            REZONATOR_LENGTH - TARGET_VERTICAL_OFFSET],
                        [TARGET_RIGHT_OFFSET, REZONATOR_LENGTH - TARGET_VERTICAL_OFFSET],
                        [TARGET_RIGHT_OFFSET, REZONATOR_LENGTH - TARGET_VERTICAL_OFFSET - TARGET_WIDTH]])

    # матрица cмещения на TARGETS_STEP мм влево
    target_move_matrix = np.array([[1, 0, -TARGETS_STEP],
                                [0, 1, 0],
                                [0, 0, 1]])
    # вторая ветка
    target2 = np.dot(target_move_matrix, target1)

    # рабочая область
    working_area = np.array([[-WORKING_AREA_WIDTH_HALF, WORKING_AREA_Y_OFFSET],
                            [WORKING_AREA_WIDTH_HALF, WORKING_AREA_Y_OFFSET],
                            [WORKING_AREA_WIDTH_HALF,
                                WORKING_AREA_Y_OFFSET + WORKING_AREA_HEIGHT],
                            [-WORKING_AREA_WIDTH_HALF,
                                WORKING_AREA_Y_OFFSET + WORKING_AREA_HEIGHT],
                            [-WORKING_AREA_WIDTH_HALF, WORKING_AREA_Y_OFFSET]])

    # Запрещенная область
    forbidden_area = np.array([[0, FORBIDDEN_AREA_DOWN_OFFSET],
                            [REZONATOR_HEIGHT_HALF - FORBIDDEN_AREA_SIDE_OFFSET,
                                FORBIDDEN_AREA_DOWN_OFFSET],
                            [REZONATOR_HEIGHT_HALF - FORBIDDEN_AREA_SIDE_OFFSET,
                                REZONATOR_LENGTH - FORBIDDEN_AREA_UP_OFFSET],
                            [REZONATOR_HEIGHT_HALF - BRANCH_WIDTH + FORBIDDEN_AREA_SIDE_OFFSET,
                                REZONATOR_LENGTH - FORBIDDEN_AREA_UP_OFFSET],
                            [REZONATOR_HEIGHT_HALF - BRANCH_WIDTH + FORBIDDEN_AREA_SIDE_OFFSET,
                                REZONATOR_LENGTH - BRANCH_LENGTH - FORBIDDEN_AREA_SIDE_OFFSET],
                            [-REZONATOR_HEIGHT_HALF + BRANCH_WIDTH - FORBIDDEN_AREA_SIDE_OFFSET,
                                REZONATOR_LENGTH - BRANCH_LENGTH - FORBIDDEN_AREA_SIDE_OFFSET],
                            [-REZONATOR_HEIGHT_HALF + BRANCH_WIDTH - FORBIDDEN_AREA_SIDE_OFFSET,
                                REZONATOR_LENGTH - FORBIDDEN_AREA_UP_OFFSET],
                            [-REZONATOR_HEIGHT_HALF + FORBIDDEN_AREA_SIDE_OFFSET,
                                REZONATOR_LENGTH - FORBIDDEN_AREA_UP_OFFSET],
                            [-REZONATOR_HEIGHT_HALF + FORBIDDEN_AREA_SIDE_OFFSET,
                                FORBIDDEN_AREA_DOWN_OFFSET],
                            [0, FORBIDDEN_AREA_DOWN_OFFSET]])
    
    return {
        'rezonator': rezonator,
        'targets': [target1, target2],
        'working_area': working_area,
        'forbidden_area': forbidden_area
    }


if __name__ == '__main__':
    rezonator = rezonator_as_object()
    
    # output as JSON to stdout
    import json
    from misc.common import NumpyArrayEncoder

    print(json.dumps(rezonator, cls=NumpyArrayEncoder))
