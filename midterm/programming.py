#!/usr/bin/env python3

import scipy
import numpy as np
from scipy.interpolate import barycentric_interpolate, CubicSpline 
import matplotlib.pyplot as plt
from scipy.special import gamma

# original function: 1/(1+x^2)
# 3 sample method: equal_space, chebyshev, uniform
# 3 interpolation method: Cubic Spline, Barycentric(kit), AAA
# implement the error function
Method = ['cubic_spline', 'barycentric', 'aaa']
Sample_Size = 50
# Func = 'FIG52'
Func = 'COS'

class Given:
    def __init__(self, _sample_method):
        self.intv_left, self.intv_right = -1, 1
        # self.intv_left, self.intv_right = -2*np.pi, 2*np.pi
        # self.func = lambda _x: 1 / (1+25*(_x**2))
        self.func = lambda _x: np.cos(10*_x)
        # self.func = lambda _x: np.log(2+_x**4)/(1-16*_x**4)
        if _sample_method == 'equal_space':
            self.sample_method = lambda _size: np.linspace(self.intv_left, self.intv_right, _size)
        elif _sample_method == 'uniform':
            self.sample_method = lambda _size: np.sort(np.random.uniform(self.intv_left, self.intv_right, size=_size))
        elif _sample_method == 'chebyshev':
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
    def aaa(self):
        # initial 
        target_err = 1e-14
        self.w = np.array([]) # weights
        self.s = [] # set of current supporting points
        r = np.ones(self.size) + np.mean(self.y) # output of remaining points
        _new_pt = np.argmax(np.abs(self.y-r))
        # greedy
        for i in range(int(self.size/2)):
            self.s.append(_new_pt)
            # Cauchy Matrix
            s_c = np.delete(np.arange(self.size), self.s)
            C = 1 / (self.x[s_c, None] - self.x[self.s][None, :])
            SF = np.diag(self.y[s_c])
            Sf = np.diag(self.y[self.s])
            A = np.dot(SF, C) - np.dot(C, Sf)
            # Solve SVD
            _, _, Vh = np.linalg.svd(A, full_matrices = False)
            self.w = Vh[-1, :].conj()
            r = self.aaa_algorithm(self.x)
            # pick the new remaining points
            err = np.abs(self.y-r)
            _new_pt = np.argmax(np.abs(err))
            max_err = err[_new_pt]
            if max_err < target_err:
                if (self.size % 10 == 0):
                    print("Sample size:", self.size, ", m:", len(self.s))
                break
        return self.aaa_algorithm
    def aaa_algorithm(self, _xs):
        ret = []
        # rational barycentric
        for _x in _xs:
            if _x in self.x[self.s]:
                _i = np.where(self.x == _x)
                ret.append(self.y[_i][0])
            else:
                num = np.sum((self.w*self.y[self.s])/(_x-self.x[self.s]))
                dom = np.sum(self.w/(_x-self.x[self.s]))
                ret.append(num/dom)
        return ret

def plot(func_1, func_2, _intl, _sample, _func):
    _x = np.linspace(-1, 1, 1000)
    _y1 = func_1(_x)
    _y2 = func_2(_x)

    # plt
    plt.clf()
    plt.plot(_x, _y1, 'o', color='r', label='sampling')
    plt.plot(_x, _y2, '-', color='g', label='fitting polynomial')
    plt.title(_sample+' on '+_intl)
    ax = plt.gca()
    plt.legend()
    plt.savefig('_result/'+_sample+'_'+_intl+'_'+_func+'.png')
    plt.show()

def maximum_error(left, right, func_1, func_2):
    _x = np.linspace(left, right, 10000)
    _y1 = func_1(_x)
    _y2 = func_2(_x)
    return np.max(np.abs(_y1-_y2))

def sample_interpolate(_sample, _intl, _range):
    _touch = False
    _ret = []
    given = Given(_sample)
    for sample_size in _range:
        # get samples from original function
        x_samples = given.sample(sample_size)
        y_samples = given.func(x_samples)
        interpolate = Interpolate(x_samples, y_samples)
        intl_func = interpolate.method(_intl)

        # calculate the max error
        _left, _right = given.interval()
        _err = maximum_error(_left, _right, given.func, intl_func)
        _ret.append(_err)
        
        if sample_size == 40:
            # plt
            plot(given.func, intl_func, _intl, _sample, Func)


    return _ret

_range = range(11, 100)
# _sample_method = 'equal_space'
_sample_method = 'uniform'
# _sample_method = 'chebyshev'
_e1 = sample_interpolate(_sample_method, 'aaa', _range)
_e2 = sample_interpolate(_sample_method, 'barycentric', _range)
_e3 = sample_interpolate(_sample_method, 'cubic_spline', _range)

plt.clf()
plt.plot(_range, np.log10(_e1), '-', color='r', label='aaa')
plt.plot(_range, np.log10(_e2), '-', color='g', label='barycentric')
plt.plot(_range, np.log10(_e3), '-', color='b', label='cubic_spline')
plt.title('sample on '+_sample_method)
plt.xlabel("N + 1 nodes")
plt.ylabel("maximum err (log with base 10)")
plt.legend()
plt.savefig('_result/'+_sample_method+'_all_'+Func+'.jpg')
plt.show()
