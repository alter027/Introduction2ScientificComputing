import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# P1: Solve u''=f with u(0)=0, u(1)=0
def f1(x):
    if x >= 0.4 and x <= 0.6:
        return 1.0
    return 0.0
def f1_exact(x):
    y = []
    for _x in x:
        if _x < 0.4:
            _y = -0.1*_x
        elif _x <= 0.6:
            _y = 0.5*(_x**2) - 0.5*_x + 0.08
        else:
            _y = 0.1*_x -0.1
        y.append(_y)
    return y

# P2: Solve u''-2u'+u=1 with u(0)=0, u'(1)=1
def f2(x):
    return 1.0
def f2_exact(x):
    e = np.exp(1)
    y = []
    for _x in x:
        _y = (((e+1)/(2*e)*_x)-1)*np.exp(_x)+1
        y.append(_y)
    return y

# P3: Solve u''=sin(2pi*x) with u'(0)=0, u'(1)=0
def f3(x):
    return np.sin(2*np.pi*x)
def f3_exact(x):
    y = []
    for _x in x:
        _y = -np.sin(2*np.pi*_x)/(4*(np.pi**2))+_x/(2*np.pi)
        y.append(_y)
    return y

# P4: Solve u''= e^sin(x) with u'(0)=0, u'(1)=a
def f4(x):
    return np.exp(np.sin(x))
def f4_exact(x, C2=0):
    # Create a fine grid for integration
    x_fine = np.linspace(0, 1, 2000)
    f_vals = f4(x_fine)
    # First integral: u'(x) = ∫₀ˣ e^(sin(t)) dt
    u_prime_fine = cumulative_trapezoid(f_vals, x_fine, initial=0)
    # Second integral: u(x) = ∫₀ˣ u'(s) ds + C₂
    u_fine = cumulative_trapezoid(u_prime_fine, x_fine, initial=0) + C2
    # Interpolate to requested points
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x_fine, u_fine)

    return cs(x)

from scipy.integrate import cumulative_trapezoid

# P5: Solve eu''+(1+e)u'+u=0 with e=0.01 u(0)=0, u(1)=1
def f5(x):
    return 0.0
def f5_exact(x):
    e = np.exp(1)
    y = []
    for _x in x:
        _y = np.exp(1-_x)-np.exp(1-100*_x)
        y.append(_y)
    return y