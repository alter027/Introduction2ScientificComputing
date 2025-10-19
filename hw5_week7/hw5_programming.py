#!/usr/bin/env python3

import scipy
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# 2 equation: y'=-5y, y'=y-y^2 
# 2 numerical method: forward_euler, backward_euler
# stability

f1DE = lambda t, y: -5*y
f1Solved = lambda t, y0: np.exp(-5*t)
f2DE = lambda t, y: y*(1-y)
f2Solved = lambda t, y0: np.exp(t) / (np.exp(t)+((1-y0)/y0))

class Numerical:
    def __init__(self, fDE, fSolved, start, end, y0):
        self.fDE = fDE
        self.fSolved = fSolved
        self.start = start
        self.end = end
        self.y0 = y0
    def method(self, method, h):
        return getattr(self, method)(h)
    def forward_euler(self, h):
        steps = int((self.end-self.start+h) / h)
        t = np.linspace(self.start, self.start + (steps-1)*h, steps)
        u = np.zeros(steps)
        u[0] = self.y0
        for i in range(1, steps):
            u[i] = u[i-1] + h*self.fDE(t[i-1], u[i-1])
        return t, u, self.fSolved(t, self.y0)
    def backward_euler(self, h):
        steps = int((self.end-self.start+h) / h)
        t = np.linspace(self.start, self.start + (steps-1)*h, steps)
        u = np.zeros(steps)
        u[0] = self.y0
        for i in range(1, steps):
            relation = lambda _u: _u - u[i-1] - h*self.fDE(t[i], _u)
            u[i] = fsolve(relation, u[i-1])[0]
        # fsolve: Powell hybrid method
        # An alternative, direction calculate the function since f is linear
        return t, u, self.fSolved(t, self.y0)

# problem 1
_func = 'diff_y=-5y'
for step in (0.1, 0.4, 0.41):
    for _method in ('forward_euler', 'backward_euler'):
        numerical = Numerical(f1DE, f1Solved, 0, 10, 1)
        _t, _u, _y = numerical.method(_method, step)

        # plt
        plt.clf()
        plt.plot(_t, _u, '-', color='r', label=_method)
        plt.plot(_t, _y, '-', color='g', label='exact')
        plt.xlabel("t")
        plt.ylabel("y")
        plt.legend()
        plt.title(_func+'_'+_method+'_h='+str(step))
        plt.savefig('_result/'+_func+'_'+_method+'_'+str(int(step*100))+'.png')
        # plt.show()

# problem 2
_func = 'diff_y=y-y^2'
for y0 in (0.9,):
    plt.clf()
    for step in (1.2,):
        _method = 'forward_euler'
        numerical = Numerical(f2DE, f2Solved, 0, 100, y0)
        _t, _u, _y = numerical.method(_method, step)

        # plt
        plt.plot(_t, _u, '-', label='h='+str(step))
    plt.plot(_t, _y, '-', color='r', label='exact')
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title(_func+'_'+_method+'_y0='+str(y0)+'_h='+str(step))
    plt.legend()
    plt.savefig('_result/'+_func+'_'+_method+'_y0='+str(y0)+'_'+str(int(step*100))+'.png')
    plt.show()
