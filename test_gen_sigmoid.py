#!/usr/bin/env python

from matplotlib import pyplot as plt
import numpy as np

from misc.common import gen_sigmoid

if __name__ == '__main__':
    """Plot sigmoid function"""
    x = np.linspace(0, 2, 100)
    y = [gen_sigmoid(k=5.0, x_offset_to_right=0.2)(xp) for xp in x]
    plt.plot(x, y)
    plt.grid()
    plt.show()
