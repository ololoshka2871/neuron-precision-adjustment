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
# Дополнительно: (Нельзя ставить 0)
#    - Смещение резонатора по X [mm]
#    - Смещение резонатора по Y [mm]
#    - Угол наклона резонатора [*]
#    - Цель измнения частоты [Hz]
FITNES_WEIGHTS = (-3.0, -0.25, -10.0, -0.25, 0.5, -0.05, 0.5, 0.5, 1e-10, 1e-10, 1e-10, 1e-10)

F_HISTORY_SIZE = 10
LASER_POWER = 30.0  # [W]

POWER_THRESHOLD = 0.05
DEST_FREQ_CH = 50.0
MAX_T = 100.0

SIM_TIMEOUT = 10.0

total_parameters = NNController.init_model(F_HISTORY_SIZE)

creator.create("FitnessMax", base.Fitness, weights=FITNES_WEIGHTS)

# Создать класс Individual, наследованный от array.array, но содержащий поля
# fitness, которое будет содержать объект класса FitnessMax определенного выше
# и typecode, который будет 'f
creator.create("Individual", array.array, 
               typecode='f',
               fitness=creator.FitnessMax) # type: ignore

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_float", random.random)

# Structure initializers
toolbox.register("individual", lambda: creator.Individual(NNController.shuffled_weights()))  # type: ignore
toolbox.register("population", tools.initRepeat, list,
                 toolbox.individual)  # type: ignore


def eval_rezonator_adjust(individual): 
    SIM_CYCLE_TIME = 0.05

    sim = Simulator(RezonatorModel(power_threshold=POWER_THRESHOLD),
                    NNController(individual), (-100, 15),
                    freq_history_size=F_HISTORY_SIZE)

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
                                  k=1.0 / LASER_POWER, x_offset_to_right=1),
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
            return {
                'grade': grade,
                'sim_offset': offset,
                'sim_angle': angle,
                'sim_def_freq': def_freq
            }


toolbox.register("evaluate", eval_rezonator_adjust)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, sigma=0.3, mu=0.0, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)


def learn_main(polulation_size: int, n_gen: int, checkpoint_file: str, 
               multyprocess: bool | None = None, gens_for_checkpoint = 1, verbose = True, 
               cxpb=0.5, mutpb=0.2):
    import os
    import pickle

    multyprocess = multyprocess if multyprocess else os.name == 'nt'

    if multyprocess:
        import multiprocessing
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

    stats = tools.Statistics(lambda ind: ind.fitness.values)  # type: ignore
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    try:
        with open(checkpoint_file, "rb") as cp_file:
            cp = pickle.load(cp_file)  # type: ignore
        population = cp["population"]
        start_gen = cp["generation"]
        hof = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
    except FileNotFoundError:
        # Start a new evolution
        population = toolbox.population(n=polulation_size)  # type: ignore
        start_gen = 0
        hof = tools.HallOfFame(maxsize=1)  # Здесь будут геномы самых лучших представителей каждого поколения
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])  # type: ignore

    for gen in range(start_gen, n_gen):
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)  # type: ignore
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit['grade']
            # TODO: params of sim

        hof.update(population)
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        population = toolbox.select(population, k=len(population))  # type: ignore
        population = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)

        if gen % gens_for_checkpoint == 0:
            print("Save state...")
            
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen, halloffame=hof,
                      logbook=logbook, rndstate=random.getstate())

            with open(checkpoint_file, "wb") as cp_file:
                pickle.dump(cp, cp_file)

    if multyprocess:
        pool.close()  # type: ignore

    return population, stats, hof

if __name__ == '__main__':
    POPULATION_SIZE = 5
    N_GEN = 100

    learn_main(POPULATION_SIZE, N_GEN, checkpoint_file='learn_v2.ckl', gens_for_checkpoint=1)