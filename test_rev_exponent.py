#!/usr/bin/env python

from matplotlib import pyplot as plt
import numpy as np

from misc.common import gen_sigmoid

if __name__ == '__main__':
    """Plot rev exponent function"""
    x = np.linspace(0, 1, 100)
    f = gen_sigmoid(A=2.0, k=5.0, x_offset_to_right=0, vertical_shift=-1.0)
    y = [f(xp) for xp in x]
    plt.plot(x, y)
    plt.grid()
    plt.show()
