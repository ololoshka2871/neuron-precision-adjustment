#!/usr/bin/env python

from matplotlib import pyplot as plt
import numpy as np

from misc.common import normal_dist

if __name__ == '__main__':
    """Plot normal function"""
    x = np.linspace(0, 1, 100)
    y = [normal_dist(xp, mean=0.3, sd=0.10) for xp in x]
    plt.plot(x, y)
    plt.grid()
    plt.show()
