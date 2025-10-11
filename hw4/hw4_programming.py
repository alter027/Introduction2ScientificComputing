#!/usr/bin/env python3

import scipy
import numpy as np
from scipy.integrate import quad, trapezoid
from scipy.misc import derivative
from pynverse import inversefunc

class Given:
    def __init__(self, _func):
        if _func == 'func_1':
            self.func = lambda _x: 1 / (1+25*(_x**2))
        elif _func == 'func_2':
            self.func = lambda _x: np.log(_x) / (1+25*(_x**2))
        else: print('Unknown Function')
    def function(self):
        return self.func

# Question 1: int_0^inf 1/(1+25*x^2)
given = Given('func_1')
result = quad(given.function(), 0, np.inf)
print("Q1: val:", result[0], ", err:", result[0]-np.pi/10)
# Q1: val: 0.31415926535897887 , abs: -4.440892098500626e-16

# Question 2:
diff_func = lambda _x: (_x*25*(_x**2)) / (1+25*(_x**2))
left_diff = diff_func(np.exp(-10))*11
left_result = -np.exp(-10)*11
print(left_diff, left_result)

given = Given('func_2')
_func = given.function()
left = np.exp(-10)
_x = np.linspace(left, 1, 10**7)
_y = _func(_x)
right_diff = 0
for i in range(10**7):
    cur_x = _x[i]
    second_derivative = derivative(_func, cur_x, n=2, dx=1e-7)
    if (i%10**6 == 0):
        print(cur_x, second_derivative, (second_derivative*(1-left)**3)/12)
    right_diff += abs((second_derivative*1e-21)/12)
right_result = trapezoid(_y, x=_x)
print(right_result, right_diff)
print("Q2: val:", left_result+right_result, ", err:", left_diff+right_diff)
# Q2: val: -0.5454445634462095 , abs: 4.4109912383427195e-11
