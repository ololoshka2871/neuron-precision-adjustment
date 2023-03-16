LASER_POWER = 30.0  # [W]
F_HISTORY_SIZE = 10
MOVE_HISTORY_SIZE = 10
POWER_THRESHOLD = 0.05
DEST_FREQ_CH = 50.0
MAX_T = 100.0
MAX_F = 500.0
MIN_AVG_SPEED = 0.1
FREQMETER_PERIOD = 0.4  # [s]

START_ENERGY = 1.0
ENERGY_CONSUMPTION_PRE_1 = 0.15
ENERGY_FIXED_TAX = 0.05
ENERGY_INCOME_PER_HZ = 1.0

SIM_CYCLE_TIME = 0.005
SIM_TIMEOUT = 10.0

# Веса оценок работы симуляции
# Оценка:
#    - Относительная дистанция до целевой частоты - меньше - лучше
#    - Относительный штраф за попадание куда не надо - меньше - лучше
#    - Относителдьный диссбаланс - меньше - лучше
#    - Точность самооценки - больше - лучше
#    - Максимальная достигнутая температура - меньше - лучше
#    - Средняя скорость движения - больше - лучше
#    - Относительное время симуляции - меньше - лучше
#    - Бонус за дину пройденнго пути - больше - лучше
#    - Бонус за остаток энергии - больше - лучше
#    - Оценка за причину остановки - больше - лучше
FITNES_WEIGHTS = [-15.0, -10.0, -1.5, 0.25, -0.05, 0.5, -0.25, 0.5, 0.5, 0.5]