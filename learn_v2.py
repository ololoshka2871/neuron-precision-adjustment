#!/usr/bin/env python

import random

import numpy as np

from deap import algorithms, base, creator, tools

from misc.common import Rezonator, gen_sigmoid
from controllers.controller_v2 import NNController
from graders.controller_grader_v2 import ControllerGrager
from misc.coordinate_transformer import CoordinateTransformer, WorkzoneRelativeCoordinates
from misc.f_s_transformer import FSTransformer
from models.rezonator_model import RezonatorModel
from models.sim_stop_detector_v2 import SimStopDetector
from simulators.simulator_v2 import Simulator

from deap_elements.fitnes_max import register_finex_max
from deap_elements.individual import register_individual


# Веса оценок работы симуляции
# Оценка:
#    - Относительная дистанция до целевой частоты - меньше - лучше
#    - Относителдьный диссбаланс - меньше - лучше
#    - Относительный штраф за попадание куда не надо - меньше - лучше
#    - Относительное время симуляции - меньше - лучше
#    - Точность самооценки - больше - лучше
#    - Максимальная достигнутая температура - меньше - лучше
#    - Средняя скорость движения - больше - лучше
#    - Оценка за причину остановки - больше - лучше
#FITNES_WEIGHTS = [-5.0, -0.25, -10.0, -0.25, 0.5, -0.05, 0.5, 0.5]
FITNES_WEIGHTS = [-5.0, -0.25, -10.0, 0.0, 0.8, 0.0, 0.2, 0.0]

F_HISTORY_SIZE = 10
MOVE_HISTORY_SIZE = 10


NNController.init_model(F_HISTORY_SIZE, MOVE_HISTORY_SIZE)

register_finex_max()
register_individual(creator.FitnessMax)  # type: ignore

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_float", random.random)

# Structure initializers
toolbox.register("individual", 
                 lambda: creator.Individual(NNController.shuffled_weights()))  # type: ignore
toolbox.register("population", tools.initRepeat, list,
                 toolbox.individual)  # type: ignore


def eval_rezonator_adjust(individual):
    SIM_CYCLE_TIME = 0.01
    MAX_F = 1000.0
    LASER_POWER = 30.0  # [W]
    POWER_THRESHOLD = 0.05
    DEST_FREQ_CH = 50.0
    MAX_T = 100.0

    SIM_TIMEOUT = 10.0

    rezonator_model = RezonatorModel(power_threshold=POWER_THRESHOLD)
    initial_pos = WorkzoneRelativeCoordinates(0.0, 1.0)
    rez = Rezonator.load()

    # Генерируем случайное смещение и случайный угол поворота
    offset = (np.random.random() * 0.3, np.random.random() * 0.5)
    angle = np.random.random() * 20 - 10
    coord_transformer = CoordinateTransformer(rez, (0, 0), offset, angle)

    controller = NNController(individual)

    sim = Simulator(rezonator_model=rezonator_model,
                    controller_v2=controller,
                    coord_transformer=coord_transformer,
                    fs_transformer=FSTransformer(255.0, MAX_F),
                    laser_power=LASER_POWER,
                    modeling_period=SIM_CYCLE_TIME,
                    freq_history_size=F_HISTORY_SIZE,
                    initial_wz_pos=initial_pos)

    stop_detector = SimStopDetector(timeout=SIM_TIMEOUT,
                                    history_len_s=0.5,
                                    min_path=0.05,
                                    min_avg_speed=0.05,
                                    min_laser_power=POWER_THRESHOLD * 0.5,
                                    max_temperature=MAX_T,
                                    self_grade_epsilon=0.01,
                                    start_timestamp=0.0)

    # Случайнное смещение целевой частоты
    def_freq = np.random.normal(DEST_FREQ_CH, 10.0)
    grader = ControllerGrager(dest_freq_ch=DEST_FREQ_CH,
                              f_penalty=gen_sigmoid(
                                  k=LASER_POWER, x_offset_to_right=0.2),  # экспериментальные параметры
                              max_temperature=MAX_T,
                              grade_weights=np.array(FITNES_WEIGHTS))

    stop_condition = sim.perform_modeling(stop_detector)

    rm = rezonator_model.get_metrics()
    total, g = grader.get_grade(rm, stop_detector.summary(), stop_condition)

    return {
        'stop_condition': stop_condition,
        'fitness': total,
        'grade': g,
        'penalty': rm['penalty_energy'],
        'sim_offset': offset,
        'sim_angle': angle,
        'sim_def_freq': def_freq
    }


toolbox.register("evaluate", eval_rezonator_adjust)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, sigma=0.3, mu=0.0, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)


def learn_main(polulation_size: int, n_gen: int, checkpoint_file: str,
               multyprocess: bool | None = None, gens_for_checkpoint=1, verbose=True,
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
        gen = cp["generation"]
        hof = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
    except FileNotFoundError:
        # Start a new evolution
        population = toolbox.population(n=polulation_size)  # type: ignore
        gen = 0
        # Здесь будут геномы самых лучших представителей каждого поколения
        hof = tools.HallOfFame(maxsize=1)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])  # type: ignore

    while True:
        gen += 1
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)  # type: ignore
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit['fitness'],)
            ind.grade = fit['grade']
            ind.rezonator_offset = fit['sim_offset']
            ind.rezonator_angle = fit['sim_angle']
            ind.adjust_freq = fit['sim_def_freq']

            if ind.fitness.values[0] > 1.1:
                g = ind.grade
                print(
                    f"Found anomaly fitness ({ind.fitness.values[0]}): {fit['stop_condition']}; Fd:{g[0]:.2f}, db:{g[1]:.2f}, fzp:{g[2]:.2f} ({fit['penalty']}), t:{g[3]:.2f}, sg:{g[4]:.2f}, Tmax:{g[5]:.2f}, Va:{g[6]:.2f}, scg:{g[7]:.2f}")

        hof.update(population)
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        population = toolbox.select(population, k=len(population))  # type: ignore
        population = algorithms.varAnd(
            population, toolbox, cxpb=cxpb, mutpb=mutpb)

        if gen % gens_for_checkpoint == 0:
            print(f"Save state >>> {checkpoint_file}")

            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen, halloffame=hof,
                      logbook=logbook, rndstate=random.getstate())

            with open(checkpoint_file, "wb") as cp_file:
                pickle.dump(cp, cp_file)


if __name__ == '__main__':
    POPULATION_SIZE = 25
    N_GEN = 50

    learn_main(POPULATION_SIZE, N_GEN,
               checkpoint_file='learn_v2.ckl', gens_for_checkpoint=1)
