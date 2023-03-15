from deap import base, creator


def register_finex_max():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))