#!/usr/bin/env python

import array
import multiprocessing
import random
import time

import numpy as np

from deap import algorithms, base, creator, tools
from common import gen_sigmoid

from controller import NNController
from controller_grader import ControllerGrager
from rezonator_model import RezonatorModel
from sim_stop_detector import SimStopDetector, StopCondition
from simulator import Simulator

# Веса оценок работы симуляции
#Оценка:
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

SIM_TIMEOUT=5.0

total_parameters = NNController.init_model(HISTORY_SIZE)

creator.create("FitnessMax", base.Fitness, weights=FITNES_WEIGHTS)
creator.create("Individual", array.array, typecode='f',
               fitness=creator.FitnessMax)  # type: ignore

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_float", random.random)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual,  # type: ignore
                 toolbox.attr_float, total_parameters)  # type: ignore
toolbox.register("population", tools.initRepeat, list,
                 toolbox.individual)  # type: ignore


def eval_rezonator_adjust(individual):
    global processed_count

    SIM_CYCLE_TIME = 0.05

    def grade_stop_condition(sc: StopCondition) -> float:
        match sc:
            case StopCondition.TIMEOUT:
                return -0.5
            case  StopCondition.STALL:
                return -0.7
            case  StopCondition.LOW_POWER:
                return -0.6
            case  StopCondition.OVERHEAT:
                return -0.2
            case  StopCondition.SELF_STOP:
                return 0.2
            case _:
                return 0.0

    sim = Simulator(RezonatorModel(power_threshold=POWER_THRESHOLD),
                    NNController(individual), (-100, 15),
                    freq_history_size=HISTORY_SIZE)

    sim_stop_detector = SimStopDetector(timeout=SIM_TIMEOUT,
                                        history_len_s=2.5,
                                        min_avg_speed=0.05,
                                        min_laser_power=POWER_THRESHOLD * 0.8,
                                        max_temperature=MAX_T)

    grader = ControllerGrager(dest_freq_ch=DEST_FREQ_CH,
                              f_penalty=gen_sigmoid(
                                  k=1.0 / LASER_POWER, x_offset=-6),
                              max_temperature=MAX_T,
                              grade_stop_condition=grade_stop_condition)

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

        stop_condition = sim_stop_detector.tick(SIM_CYCLE_TIME, mmetrics)

        if stop_condition != StopCondition.NONE:
            print(f"Done with condition: {stop_condition}")
            return grader.get_grade(
                rmetrics, sim_stop_detector.summary(), stop_condition)


toolbox.register("evaluate", eval_rezonator_adjust)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, sigma=0.3, mu=0.0, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)

if __name__ == "__main__":
    # Process Pool of all cores
    pool = multiprocessing.Pool(processes=8)
    toolbox.register("map", pool.map)

    pop = toolbox.population(n=POPULATION_SIZE)  # type: ignore
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)  # type: ignore
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=N_GEN,
                        stats=stats, halloffame=hof)

    pool.close()
