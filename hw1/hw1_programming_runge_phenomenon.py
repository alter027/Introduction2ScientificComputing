#!/usr/bin/env python3

import scipy
import numpy as np
from scipy.interpolate import lagrange, barycentric_interpolate
import matplotlib.pyplot as plt

# 2 kinds of original function: 1/(1+x^2), sin(x)
# 5 interpolation method: Vandermonde, Newton, Lagrange(kit),
#                         Modified Lagrange, Barycentric(kit)
Method = ['vandermonde', 'newton', 'lagrange', 'lagrange_modified', 'barycentric']
# different sample size: 10, 30, 50, 100, 500, 1000
SampleSize = [10, 30, 50, 100, 500, 1000]

class Given:
    def __init__(self, _func):
        if _func == 'func_1':
            self.intv_left, self.intv_right = -5, 5
            self.func = lambda _x: 1 / (1+_x**2)
        elif _func == 'func_2':
            self.intv_left, self.intv_right = 0, 1
            self.func = lambda _x: np.sin(_x)
        else: print('Unknown Function')
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
    def vandermonde(self):
        v_matrix = np.vander(self.x, increasing=True)
        v_coef = np.linalg.solve(v_matrix, self.y)
        return lambda _x: np.polyval(v_coef[::-1], _x)
    def newton(self):
        self.nt_num = np.zeros(self.size)
        for i in range(self.size):
            if i == 0:
                self.nt_num[i] = self.y[i]
                continue
            val, dom = self.newton_value(self.x[i], i-1)
            self.nt_num[i] = (self.y[i]-val)/dom
        return self.newton_execute
    def newton_value(self, _x, degree):
        _y = self.nt_num[degree]
        nt_dom = _x-self.x[degree]
        for i in range(degree-1, -1, -1):
            _y = _y * (_x-self.x[i])
            _y = _y + self.nt_num[i]
            nt_dom = nt_dom * (_x-self.x[i])
        return _y, nt_dom
    def newton_execute(self, x_arr):
        y_arr = np.zeros(len(x_arr))
        for _i, _x in enumerate(x_arr):
            _y, _ = self.newton_value(_x, self.size-1)
            y_arr[_i] = _y
        return y_arr
    def lagrange(self):
        return lagrange(self.x, self.y)
    def lagrange_modified(self):
        self.li_dmt = np.ones(self.size)
        for i in range(self.size):
            for j in range(self.size):
                if i is j: continue
                self.li_dmt[i] = self.li_dmt[i] * (self.x[i]-self.x[j])
        return self.lagrange_modified_execute
    def lagrange_modified_execute(self, x_arr):
        y_arr = np.zeros(len(x_arr))
        for _i, _x in enumerate(x_arr):
            # indicate whether the value is including in the sampling of original function
            is_sampled = -1
            diff = np.ones(self.size)
            product = 1.0
            for i in range(self.size):
                if _x == self.x[i]: is_sampled = i 
                diff[i] = _x - self.x[i]
                product = product * diff[i]
            if is_sampled == -1:
                _y = 0.0
                for i in range(self.size):
                    _y = _y + (product * self.y[i])/(diff[i] * self.li_dmt[i])
            else:
                _y = self.y[is_sampled]
            y_arr[_i] = _y
        return y_arr
    def barycentric(self):
        return lambda _x: barycentric_interpolate(self.x, self.y, _x)
 
for _func in ['func_1', 'func_2']:
    given = Given(_func)
    intv_left, intv_right = given.interval()
    for sample_size in SampleSize:
        # get samples from original function
        x_samples = np.linspace(intv_left, intv_right, sample_size)
        y_samples = given.func(x_samples)
        # print("\n| "+str(sample_size)+" |", end='')
        for _method in Method:
            # _name = '_result/'+_func+'_'+_method+'_sample'+str(sample_size)
            # print(" <img src="+_name+".png width=\"200\" height=\"200\" />"+" |", end="")
            print('======', _func, sample_size, _method, '======')
            # form the interpolate function
            interpolate = Interpolate(x_samples, y_samples)
            intl_func = interpolate.method(_method)

            # get the points for plt
            x_plt = np.arange(intv_left, intv_right, 0.005)
            x_plt = np.sort(np.append(x_plt, x_samples))
            y_plt_intl = intl_func(x_plt)
            y_plt_func = given.func(x_plt)

            # plt
            plt.clf()
            plt.plot(x_samples, y_samples, 'o', color='r', label='sampling')
            plt.plot(x_plt, y_plt_intl, '-', color='g', label='fitting polynomial')
            # plt.plot(x_plt, y_plt_func, '-', color='b', label='original function')
            plt.title(_method+'_sample'+str(sample_size))
            ax = plt.gca()
            ax.set_xlim(intv_left, intv_right)
            ax.set_ylim([-0.5, 1.5])
            plt.legend()
            plt.savefig('_result/'+_func+'_'+_method+'_sample'+str(sample_size), dpi=96)
            # plt.show()

