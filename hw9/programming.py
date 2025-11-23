import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from local_kit import solve_bvp_fdm, plot_bvp
from given_func import *

Ns = [20, 40, 80, 120, 160,] # Number of interior points
# Ns = [20, 40] # Number of interior points

# P1: Solve u''=f with u(0)=0, u(1)=0
a, ya, b, yb = 0, 0, 1, 0
for N in Ns:
    x_sol, y_sol = solve_bvp_fdm(f1, a, b, N, ya=ya, yb=yb)
    y_exact = f1_exact(x_sol)
    max_error = np.max(abs(y_sol-y_exact))
    print(f"P1: node={N:3d}, max error={max_error:.4e}")
    plot_bvp(x_sol, y_sol, y_exact, "p1_u''=f", N)

# P2: Solve u''-2u'+u=1 with u(0)=0, u'(1)=1
p2_f = lambda x: -2
q2_f = lambda x: 1
for N in Ns:
    x_sol, y_sol = solve_bvp_fdm(f2, 0, 1, N, ya=0, d_yb=1, p_f=p2_f, q_f=q2_f)
    y_exact = f2_exact(x_sol)
    max_error = np.max(abs(y_sol-y_exact))
    print(f"P2: node={N:3d}, max error={max_error:.4e}")
    plot_bvp(x_sol, y_sol, y_exact, "p2_u''-2u'+u=1", N)

# P3: Solve u''=sin(2pi*x) with u'(0)=0, u'(1)=0
for N in Ns:
    x_sol, y_sol = solve_bvp_fdm(f3, 0, 1, N, d_ya=0, d_yb=0)
    y_exact = f3_exact(x_sol)
    y_diff = y_exact[0]-y_sol[0]
    max_error = np.max(abs(y_sol+y_diff-y_exact))
    print(f"P3: node={N:3d}, max error={max_error:.4e}")
    plot_bvp(x_sol, y_sol+y_diff, y_exact, "p3_u''=sin(2pix)", N)

# P4: Solve u''= e^sin(x) with u'(0)=0, u'(1)=a
# α = ∫₀¹ e^(sin(x)) dx
alpha, _ = quad(f4, 0, 1)
print('alpha =', alpha)
for N in Ns:
    x_sol, y_sol = solve_bvp_fdm(f4, 0, 1, N, d_ya=0, d_yb=alpha)
    y_exact = f4_exact(x_sol)
    y_diff = y_exact[0]-y_sol[0]
    max_error = np.max(abs(y_sol+y_diff-y_exact))
    print(f"P4: node={N:3d}, max error={max_error:.4e}")
    plot_bvp(x_sol, y_sol+y_diff, y_exact, "p4_u''= e^sin(x)", N)

# P5: Solve eu''+(1+e)u'+u=0 with e=0.01 u(0)=0, u(1)=1
p2_f = lambda x: 101
q2_f = lambda x: 100
for N in Ns:
    x_sol, y_sol = solve_bvp_fdm(f5, 0, 1, N, ya=0, yb=1, p_f=p2_f, q_f=q2_f)
    y_exact = f5_exact(x_sol)
    max_error = np.max(abs(y_sol-y_exact))
    print(f"P5: node={N:3d}, max error={max_error:.4e}")
    plot_bvp(x_sol, y_sol, y_exact, "p5_eu''+(1+e)u'+u=0", N)

