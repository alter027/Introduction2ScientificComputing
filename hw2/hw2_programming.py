#!/usr/bin/env python3

import scipy
import numpy as np
from scipy.interpolate import barycentric_interpolate, CubicSpline 
import matplotlib.pyplot as plt

# original function: 1/(1+x^2)
# 2 sample method: equal_space, chebyshev
# 2 interpolation method: Cubic Spline, Barycentric(kit)
# implement the error function
Method = ['cubic_spline', 'barycentric']


class Given:
    def __init__(self, _func, _sample_method):
        self.intv_left, self.intv_right = -1, 1
        self.func = lambda _x: 1 / (1+25*(_x**2))
        if _sample_method == 'equal_space':
            self.sample_method = lambda _size: np.linspace(self.intv_left, self.intv_right, _size)
        elif _sample_method == 'chebyshev': #
            self.sample_method = lambda _size: np.sort(np.cos((np.arange(_size)+0.5)*np.pi/_size))
        else: print('Unknown Function')
    def sample(self, _size):
        return self.sample_method(_size)
    def func(self, x):
        return self.func(x)
    def interval(self):
        return self.intv_left, self.intv_right

class Interpolate:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = len(self.x)
    def method(self, method):
        self.func = getattr(self, method)()
        return self.func
    def cubic_spline(self):
        return CubicSpline(self.x, self.y)
    def barycentric(self):
        return lambda _x: barycentric_interpolate(self.x, self.y, _x)

def maximum_error(func_1, func_2):
    _x = np.linspace(-1, 1, 10000)
    _y1 = func_1(_x)
    _y2 = func_2(_x)
    return np.max(np.abs(_y1-_y2))

def sample_interpolate(_sample, _intl):
    _touch = False
    _ret = []
    given = Given('runge', _sample)
    for sample_size in range(100, 1600):
        # get samples from original function
        x_samples = given.sample(sample_size)
        y_samples = given.func(x_samples)
        interpolate = Interpolate(x_samples, y_samples)
        intl_func = interpolate.method(_intl)

        # calculate the max error
        _err = maximum_error(given.func, intl_func)
        _ret.append(_err)
        if not _touch and _err < 1e-10:
            print(_sample, _intl, "N + 1 =", sample_size)
            _touch = True
    return _ret

_e1 = sample_interpolate('equal_space', 'cubic_spline')
_e2 = sample_interpolate('chebyshev', 'barycentric')

# plt
_x = range(100, 1600)
plt.plot(_x, np.log10(_e1), '-', color='r', label='equal_space_cubic_spline')
plt.plot(_x, np.log10(_e2), '-', color='g', label='chevyshev_barycentric')
plt.xlabel("N + 1 nodes")
plt.ylabel("maximum err (log with base 10)")
plt.legend()
plt.savefig('_result/tri.jpg')
plt.show()

