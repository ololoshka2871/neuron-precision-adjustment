import numpy as np

from misc.common import gen_sigmoid, my_normal

LASER_POWER = 30.0  # [W]
F_HISTORY_SIZE = 10
MOVE_HISTORY_SIZE = 10
POWER_THRESHOLD = 0.05
MAX_T = 1500.0
MAX_F = 500.0
FREQMETER_PERIOD = 0.4  # [s]
AMBIENT_T = 20.0

SIM_CYCLE_TIME = 0.005
SIM_TIMEOUT = 10.0
FREQ_PRECISION = 1.0
NN_MEAN_LAYERS = 3


f_penalty = gen_sigmoid(k=LASER_POWER, x_offset_to_right=0.2,
                        vertical_shift=-0.00247262315663477434)


def gen_sim_parameters() -> dict:
    return dict(
        offset=(np.random.random() * 0.3, np.random.random() * 0.5),
        angle=np.random.random() * 20 - 10,
        initial_freq_diff=(my_normal() + 0.55) * 0.5,
        ag_thikness=my_normal() * 0.0002 + 0.0005,
    )
