import array

from deap import creator


def register_individual(fitnes_max):
    """
    Создать класс Individual, наследованный от array.array, но содержащий поля
    fitness, которое будет содержать объект класса FitnessMax определенного выше
    и typecode, который будет 'f'
    """
    creator.create("Individual", array.array,
                typecode='f',
                fitness=fitnes_max,
                grade=0.0,
                rezonator_offset=(0.0, 0.0),
                rezonator_angle=0.0,
                ag_layer_thikness=0.0,
                initial_freq_diff=0.0)