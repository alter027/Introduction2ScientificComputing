import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

def solve_fd4_direct(n):
    x = np.linspace(0, 1, n+2)
    h = x[1] - x[0]
    N = n
    f = np.exp(np.sin(x[1:-1]))
    ab = np.zeros((5, N))
    # 4th-order FD stencil for interior points
    ab[0, 2:] = -1
    ab[1, 1:] = 16
    ab[2, :] = -30
    ab[3, :-1] = 16
    ab[4, :-2] = -1
    ab = ab / (12 * h ** 2)
    # Use FD2 for first two and last two interior points
    for k in range(2):
        ab[:, k] = 0
        ab[2, k] = -2 / (h ** 2)
        ab[1, k] = ab[3, k] = 1 / (h ** 2)
        ab[:, N - 1 - k] = 0
        ab[2, N - 1 - k] = -2 / (h ** 2)
        ab[1, N - 1 - k] = ab[3, N - 1 - k] = 1 / (h ** 2)
    u_inner = solve_banded((2, 2), ab, -f)
    u = np.zeros(N + 2)
    u[1:-1] = u_inner
    return x, u

# Mesh refinements and error estimation
ns = [32, 64, 128, 256, 512, 1024]
errors = []
max_estimated_errors = []
prev_x, prev_u = None, None

for n in ns:
    x, u = solve_fd4_direct(n)
    if prev_u is not None:
        # Interpolate previous solution for error estimation
        u_interp = np.interp(x, prev_x, prev_u)
        error = np.linalg.norm(u - u_interp, ord=np.inf)
        errors.append(error)
        # Truncation error estimate from second differences
        second_diff = np.abs(u[2:] - 2*u[1:-1] + u[:-2]) / (x[1] - x[0])**2
        trunc_error_estimate = np.max(second_diff) * (x[1] - x[0])**4 / 90
        max_estimated_errors.append(trunc_error_estimate)
    prev_x, prev_u = x, u

plt.figure()
for idx, n in enumerate([ns[0], ns[-1]]):
    x, u = solve_fd4_direct(n)
    plt.plot(x, u, label=f'n={n}')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Solution to -u\"=exp(sin(x)), FD4 scheme')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.loglog(ns[1:], errors, 'o-', label='Error by mesh refinement')
plt.loglog(ns[1:], max_estimated_errors, 'x--', label='Estimated truncation error')
plt.xlabel('Grid points')
plt.ylabel('Error')
plt.title('Error and Estimate by Grid Refinement (FD4)')
plt.legend()
plt.grid()
plt.show()

print(f"Smallest observed error by grid refinement: {min(errors):.4e}")
print(f"Estimated truncation error at finest grid: {max_estimated_errors[-1]:.4e}")
print(f"Finest grid used: {ns[-1]} points")
