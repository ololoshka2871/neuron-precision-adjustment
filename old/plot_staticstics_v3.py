#!/usr/bin/env python


import argparse
import random
import pickle

import matplotlib.pyplot as plt

from deap import creator

from deap_elements.fitnes_max import register_finex_max
from deap_elements.individual import register_individual


if __name__ == '__main__':
    # parse argumants
    # optional arguments: filaname, default = 'learn_v3.ckl'
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Simulation history file', default='learn_v3.ckl')
    args = parser.parse_args()

    register_finex_max()
    register_individual(creator.FitnessMax)  # type: ignore

    # read history file
    with open(args.file, "rb") as cp_file:
        cp = pickle.load(cp_file)  # type: ignore
    
    population = cp["population"]
    hof = cp["halloffame"]
    logbook = cp["logbook"]
    random.setstate(cp["rndstate"])

    # plot statistics
    gen = logbook.select("gen")
    fit_max = logbook.select("max")
    size_avgs = logbook.select("avg")

    fig, ax1 = plt.subplots()

    line1 = ax1.plot(gen, fit_max, "r-", label="Maximum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Maximum Fitness", color="r")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, size_avgs, "b-", label="Average Fitness")
    ax2.set_ylabel("Average Fitness", color="b")

    lines = line1 + line2
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs)

    plt.show()
