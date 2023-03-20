
from misc.common import gen_sigmoid


LASER_POWER = 30.0  # [W]
F_HISTORY_SIZE = 10
MOVE_HISTORY_SIZE = 10
POWER_THRESHOLD = 0.05
DEST_FREQ_CH = 50.0
MAX_T = 1500.0
MAX_F = 500.0
MIN_AVG_SPEED = 0.1
FREQMETER_PERIOD = 0.4  # [s]
AMBIENT_T = 20.0

START_ENERGY = 1.0
ENERGY_CONSUMPTION_PRE_1 = 0.2
ENERGY_FIXED_TAX = 0.025
ENERGY_INCOME_PER_HZ = 0.25

SIM_CYCLE_TIME = 0.005
SIM_TIMEOUT = 10.0

incum_function = gen_sigmoid(k=5.0, x_offset_to_right=0.2)
f_penalty = gen_sigmoid(k=LASER_POWER, x_offset_to_right=0.2, vertical_shift=-0.00247262315663477434)
