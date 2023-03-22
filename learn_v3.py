#!/usr/bin/env python

import random
import functools

import numpy as np

from deap import algorithms, base, creator, tools
from misc.Rezonator import Rezonator

from controllers.controller_v3 import NNController
from graders.controller_grader_v3 import ControllerGrager
from misc.coordinate_transformer import CoordinateTransformer, WorkzoneRelativeCoordinates
from misc.f_s_transformer import FSTransformer
from models.rezonator_model import RezonatorModel
from models.sim_stop_detector_v3 import SimStopDetector
from models.stop_condition import StopCondition
from simulators.simulator_v3 import Simulator

from deap_elements.fitnes_max import register_finex_max
from deap_elements.individual import register_individual

from parameters_v3 import *


NNController.init_model(F_HISTORY_SIZE, MOVE_HISTORY_SIZE, mean_layers=NN_MEAN_LAYERS)

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


def eval_rezonator_adjust(individual, gen: int, it: int):
    while True:
        params = gen_sim_parameters()
        rezonator = RezonatorModel(
            power_threshold=POWER_THRESHOLD, layer_thikness=params['ag_thikness'])
        adjustment_freq = freq_from_diff(params['initial_freq_diff'])  # [Hz]
        if adjustment_freq > FREQ_PRECISION:
            break

    initial_pos = WorkzoneRelativeCoordinates(-1.0, 1.0)  # Верхний левый угол
    rez = Rezonator.load()
    coord_transformer = CoordinateTransformer(
        rez, (0, 0), params['offset'], params['angle'])

    controller = NNController(individual)

    sim = Simulator(rezonator_model=rezonator,
                    controller_v3=controller,
                    coord_transformer=coord_transformer,
                    fs_transformer=FSTransformer(255.0, MAX_F),
                    laser_power=LASER_POWER,
                    initial_freq_diff=params['initial_freq_diff'],
                    freqmeter_period=FREQMETER_PERIOD,
                    modeling_period=SIM_CYCLE_TIME,
                    freq_history_size=F_HISTORY_SIZE,
                    initial_wz_pos=initial_pos)

    stop_detector = SimStopDetector(timeout=SIM_TIMEOUT,
                                    history_len_s=SIM_TIMEOUT,
                                    max_temperature=MAX_T,
                                    self_grade_epsilon=0.01,
                                    start_timestamp=0.0)

    grader = ControllerGrager(dest_freq_ch=adjustment_freq,
                              f_penalty=f_penalty,
                              max_temperature=MAX_T)

    stop_condition = sim.perform_modeling(stop_detector)

    rm = rezonator.get_metrics()
    total, g = grader.get_grade(rm, stop_detector.summary(), stop_condition)

    return {
        'stop_condition': stop_condition,
        'fitness': total,
        'grade': g,
        'penalty': rm['penalty_energy'],
        'sim_offset': params['offset'],
        'sim_angle': params['angle'],
        'sim_ag_layer_thikness': params['ag_thikness'],
        'sim_initial_freq_diff': params['initial_freq_diff'],
    }


def eval_rezonator_adjust_wrapper(individual, gen: int, it: int):
    """
    Запустить симуляцию несколько раз для каждого генома с разными начальными условиями
    чтобы исключить фактор "угадывания"
    """
    SIM_TRYS = 4

    fitness = []
    res = dict()

    for _ in range(SIM_TRYS):
        res = eval_rezonator_adjust(individual, it=it, gen=gen)
        fitness.append(res)

    gf_only = list(map(lambda f: f['fitness'], fitness))
    best_index = gf_only.index(np.max(gf_only))
    return fitness[best_index]


toolbox.register("evaluate", eval_rezonator_adjust_wrapper)
toolbox.register("mate", tools.cxBlend, alpha=0.65)
toolbox.register("mutate", tools.mutGaussian, sigma=0.5, mu=0.0, indpb=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)


def learn_main(polulation_size: int, max_iterations: int,
               checkpoint_file: str,
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
        hof_gloabal = cp["halloffame"]
        gen_hof = cp["gen_hof"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
    except FileNotFoundError:
        # Start a new evolution
        population = toolbox.population(n=polulation_size)  # type: ignore
        gen = 0
        # Здесь будут геномы самых лучших представителей каждого поколения
        hof_gloabal = tools.HallOfFame(maxsize=2)
        gen_hof = []
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])  # type: ignore

    it = 0

    while True:
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population]  # ignore valid fitness
        fitnesses = toolbox.map(functools.partial(  # type: ignore
            toolbox.evaluate, gen=gen, it=it), invalid_ind)  # type: ignore
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit['fitness'],)
            ind.grade = fit['grade']
            ind.rezonator_offset = fit['sim_offset']
            ind.rezonator_angle = fit['sim_angle']
            ind.ag_layer_thikness = fit['sim_ag_layer_thikness']
            ind.initial_freq_diff = fit['sim_initial_freq_diff']

        best = tools.HallOfFame(maxsize=1)
        best.update(population)
        gen_hof.append(best[0])

        hof_gloabal.update(population)

        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        population = toolbox.select(  # type: ignore
            population, k=len(population) - len(hof_gloabal)) 
        population = algorithms.varAnd(
            population, toolbox, cxpb=cxpb, mutpb=mutpb)
        
        # Элитизм
        population.extend(hof_gloabal)

        if gen % gens_for_checkpoint == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen, halloffame=hof_gloabal,
                      gen_hof=gen_hof, logbook=logbook, rndstate=random.getstate())

            with open(checkpoint_file, "wb") as cp_file:
                pickle.dump(cp, cp_file)

        if max_iterations > 0 and it >= max_iterations:
            break
        
        gen += 1
        it += 1


if __name__ == '__main__':
    import argparse

    # parse argumants
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, help='Population size', default=5)
    parser.add_argument('-m', type=int, help='Max iterations', default=0)
    parser.add_argument(
        'file', type=str, help='Simulation history file', nargs='?', default='learn_v3.ckl')
    args = parser.parse_args()

    learn_main(args.p, args.m, checkpoint_file=args.file,
               gens_for_checkpoint=1)
