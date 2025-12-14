import numpy as np
import math

COS = lambda _x: np.cos(10*_x)
FA  = lambda _x: np.sqrt(1.21-_x**2)
FB  = lambda _x: np.sqrt(0.01+_x**2)
FC  = lambda _x: np.tanh(5*_x)
FD  = lambda _x: np.sin(40*_x)
FE  = lambda _x: np.exp(-1/(_x**2))

class Given:
    def __init__(self, _sample_method, _func=COS):
        self.intv_left, self.intv_right = -1, 1
        # self.intv_left, self.intv_right = -2*np.pi, 2*np.pi
        # self.func = lambda _x: 1 / (1+25*(_x**2))
        self.func = _func
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