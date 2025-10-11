#!/usr/bin/env python3

import scipy
import numpy as np
from scipy.integrate import quad, trapezoid
from scipy.misc import derivative
from pynverse import inversefunc

# 2 kinds of original function: 1/(1+x^2), sin(x)
# 5 interpolation method: Vandermonde, Newton, Lagrange(kit),
#                         Modified Lagrange, Barycentric(kit)
Method = ['vandermonde', 'newton', 'lagrange', 'lagrange_modified', 'barycentric']
# different sample size: 10, 30, 50, 100, 500, 1000
SampleSize = [10, 30, 50, 100, 500, 1000]

class Given:
    def __init__(self, _func):
        if _func == 'func_1':
            self.func = lambda _x: 1 / (1+25*(_x**2))
        elif _func == 'func_2':
            self.func = lambda _x: np.log(_x) / (1+25*(_x**2))
        else: print('Unknown Function')
    def function(self):
        return self.func

diff_func = lambda _x: (_x*25*(_x**2)) / (1+25*(_x**2))

# Question 1: int_0^inf 1/(1+25*x^2)
given = Given('func_1')
result = quad(given.function(), 0, np.inf)
print("Q1: val:", result[0], ", abs:", result[0]-np.pi/10)

# Question 2:
left_diff = diff_func(np.exp(-9))
left_result = np.exp(-9)*10
print(left_diff, left_result)

given = Given('func_2')
_func = given.function()
left = np.exp(-9)
_x = np.linspace(left, 1, 10**7)
_y = _func(_x)
# result = quad(func, 1e-50, 1)
print(left)
left_diff = 0
for i in range(10**7):
    cur_x = _x[i]
    second_derivative = derivative(_func, cur_x, n=2, dx=1e-7)
    if (i%10**6 == 0):
        print(cur_x, second_derivative, (second_derivative*(1-left)**3)/12)
    left_diff += abs((second_derivative*1e-21)/12)
result = trapezoid(_y, x=_x)
print(result, left_diff)
print("Q2: val:", result, ", abs:", left_diff)
