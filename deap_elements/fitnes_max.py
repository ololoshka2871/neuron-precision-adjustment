from deap import base, creator


def register_finex_max(weights):
    creator.create("FitnessMax", base.Fitness, weights=weights)