#!/usr/bin/env python

import array
import random

import numpy as np

from deap import algorithms, base, creator, tools
from common import gen_sigmoid

from controller import NNController
from controller_grader import ControllerGrager
from rezonator_model import RezonatorModel
from sim_stop_detector import SimStopDetector, StopCondition
from simulator import Simulator

# Веса оценок работы симуляции
# Оценка:
#    - Относительная дистанция до целевой частоты - меньше - лучше
#    - Относителдьный диссбаланс - меньше - лучше
#    - Относительный штраф за попадание куда не надо - меньше - лучше
#    - Относительное время симуляции - меньше - лучше
#    - Точность самоошенки - больше - лучше
#    - Максимальная достигнутая температура - меньше - лучше
#    - Средняя скорость движения - больше - лучше
#    - Оценка за причину остановки - больше - лучше
FITNES_WEIGHTS = (-0.5, -0.25, -1.0, -0.1, 0.2, -0.05, 0.3, 0.5)

HISTORY_SIZE = 10
POPULATION_SIZE = 100
N_GEN = 100
LASER_POWER = 30.0  # [W]
HISTORY_SIZE = 15
POWER_THRESHOLD = 0.05
DEST_FREQ_CH = 50.0
MAX_T = 100.0

SIM_TIMEOUT = 10.0

total_parameters = NNController.init_model(HISTORY_SIZE)

creator.create("FitnessMax", base.Fitness, weights=FITNES_WEIGHTS)

# Создать класс Individual, наследованный от array.array, но содержащий поля
# fitness, которое будет содержать объект класса FitnessMax определенного выше
# и typecode, который будет 'f
creator.create("Individual", array.array, 
               typecode='f',
               fitness=creator.FitnessMax)  # type: ignore

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_float", random.random)

# Structure initializers
toolbox.register("individual", lambda: creator.Individual(NNController.shuffled_weights()))  # type: ignore
toolbox.register("population", tools.initRepeat, list,
                 toolbox.individual)  # type: ignore


def eval_rezonator_adjust(individual): # individual: Individual
    SIM_CYCLE_TIME = 0.05

    sim = Simulator(RezonatorModel(power_threshold=POWER_THRESHOLD),
                    NNController(individual), (-100, 15),
                    freq_history_size=HISTORY_SIZE)

    sim_stop_detector = SimStopDetector(timeout=SIM_TIMEOUT,
                                        history_len_s=1.0,
                                        min_path=0.1,
                                        min_avg_speed=0.05,
                                        min_laser_power=POWER_THRESHOLD * 0.8,
                                        max_temperature=MAX_T)

    # Случайнное смещение целевой частоты
    def_freq = np.random.normal(DEST_FREQ_CH, 10.0)
    grader = ControllerGrager(dest_freq_ch=def_freq,
                              f_penalty=gen_sigmoid(
                                  k=1.0 / LASER_POWER, x_offset_to_right=-6),
                              max_temperature=MAX_T)

    # Генерируем случайное смещение и случайный угол поворота
    offset = (np.random.random() * 0.3, np.random.random() * 0.5)
    angle = np.random.random() * 20 - 10

    playground = sim.use_rezonator(
        RezonatorModel.get_playground, offset, angle)

    while True:
        model_pos = playground.map_to_model(sim.laser_pos())

        mmetrics = sim.tick(SIM_CYCLE_TIME, model_pos)
        rmetrics = sim.use_rezonator(RezonatorModel.get_metrics)

        # ------------ Условие останова ----------

        stop_condition = sim_stop_detector.tick(SIM_CYCLE_TIME, mmetrics, rmetrics)

        if stop_condition != StopCondition.NONE:
            grade = grader.get_grade(
                rmetrics, sim_stop_detector.summary(), stop_condition)
            print(f"Done {stop_condition}; Fd:{grade[0]:.2f}, db:{grade[1]:.2f}, pen:{grade[2]:.2f}, t:{grade[3]:.2f}, ss:{grade[4]:.2f}, Tmax:{grade[5]:.2f}, Va:{grade[6]:.2f}")
            return grade


toolbox.register("evaluate", eval_rezonator_adjust)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, sigma=0.3, mu=0.0, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)


def learn_main(polulation_size: int, n_gen: int, multyprocess = False):
    if multyprocess:
        import multiprocessing
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

    pop = toolbox.population(n=polulation_size)  # type: ignore
    hof = tools.HallOfFame(1)  # Здесь будут геномы самых лучших представителей каждого поколения
    stats = tools.Statistics(lambda ind: ind.fitness.values)  # type: ignore
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                        ngen=n_gen, stats=stats, halloffame=hof)

    if multyprocess:
        pool.close()  # type: ignore

    return pop, stats, hof

if __name__ == '__main__':
    learn_main(100, 100, multyprocess=True)