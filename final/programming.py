#!/usr/bin/env python3

import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

from interpolation import *
from given import *

# original function: 1/(1+x^2)
# 3 sample method: equal_space, chebyshev, uniform
# 3 interpolation method:
#       Cubic Spline,                                                               OK
#       Poly least-squares with oversampling (ratio=2),                             
#       Fourier extension on [-2, 2] with oversampling (ratio=2),                   
#       Fourier series plus ploy of degree sqrt(n) with oversampling (ratio=2),      
#       Floater-Hormann rational interpolation: Chebfun 'equi',                     
#       AAA (tolerance 10^-13)                                                      OK
# implement the error function
Method = [
        'cubic_spline',
        'poly_least_square',
        'fourier_extension',
        'fourier_series_poly_degree',
        'floater_hormann',
        'aaa',]
Sample_Size = 50
# Func = 'FIG52'
Func = 'COS'

def plot(func_1, func_2, _intl, _sample, _func):
    _x = np.linspace(-1, 1, 1000)
    _y1 = func_1(_x)
    _y2 = func_2(_x)
    # print(_y1, _y2)

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
    # return np.max(np.abs(_y1-_y2))
    return np.linalg.norm(_y1 - _y2, ord=np.inf)

def sample_interpolate(_sample, _intl, _range, _f):
    _touch = False
    _ret = []
    given = Given(_sample, _func=_f)
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
        
        # plot(given.func, intl_func, _intl, _sample, Func)
    return _ret

_range = range(5, 201, 5)
_sample_method = 'equal_space'
# _sample_method = 'uniform'
# _sample_method = 'chebyshev'

name_map = {FA:"FA", FB:"FB", FC:"FC", FD:"FD", FE:"FE"}
# for f in [FA, FB, FC, FD, FE]:
for f in [FC, FD, FE]:
    _e = []
    for _m in Method:
        _e.append(sample_interpolate(_sample_method, _m, _range, f))

    # print(_e)

    Color = {'cubic_spline':'blue',
            'poly_least_square':'red',
            'fourier_extension':'orange',
            'fourier_series_poly_degree':'purple',
            'floater_hormann':'g',
            'aaa':'black'}

    plt.clf()
    for _i, _m in enumerate(Method):
        _r = Color[_m]
        plt.plot(_range, np.log10(_e[_i]), '-', color=_r, label=_m)
    plt.title('sample on '+_sample_method+"_"+name_map[f])
    plt.xlabel("N + 1 nodes")
    plt.ylabel("maximum err (log with base 10)")
    plt.legend()
    plt.savefig('_result/'+_sample_method+'_all_'+name_map[f]+'.jpg')
    # plt.show()
