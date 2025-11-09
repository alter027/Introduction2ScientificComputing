import numpy as np
import matplotlib.pyplot as plt

def solve_nonlinear_bvp_fd2(N=128, u0=1.0, uN1=1.0, newton_tol=1e-10, newton_maxit=20):
    x = np.linspace(0, 1, N+2)
    h = x[1] - x[0]
    u = np.ones(N) * 0.95
    for it in range(newton_maxit):
        F = np.zeros(N)
        J = np.zeros((N, N))
        for i in range(N):
            ui = u[i]
            uim1 = u[i-1] if i > 0 else u0
            uip1 = u[i+1] if i < N-1 else uN1
            F[i] = -(uim1 - 2*ui + uip1) / h**2 + np.sin(ui)
            J[i, i] = 2/h**2 + np.cos(ui)
            if i > 0:
                J[i, i-1] = -1/h**2
            if i < N-1:
                J[i, i+1] = -1/h**2
        delta = np.linalg.solve(J, -F)
        u += delta
        if np.linalg.norm(delta, np.inf) < newton_tol:
            break
    u_full = np.zeros(N+2)
    u_full[0] = u0
    u_full[1:-1] = u
    u_full[-1] = uN1
    return x, u_full

# Mesh refinement and error estimation
Ns = [32, 64, 128, 256, 512]
solutions = []
for N in Ns:
    x, u = solve_nonlinear_bvp_fd2(N)
    solutions.append((x, u))

# Estimate error between successive grid refinements (interpolating coarse to fine)
errors = []
for k in range(1, len(Ns)):
    x_fine, u_fine = solutions[k]
    x_coarse, u_coarse = solutions[k-1]
    u_interp = np.interp(x_fine, x_coarse, u_coarse)
    error = np.linalg.norm(u_fine - u_interp, np.inf)
    errors.append(error)

# Plot u(x) for finest grid
x, u = solutions[-1]
plt.plot(x, u, label='u(x) solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title("Solution of -u'' + sin(u) = 0, u(0)=u(1)=1")
plt.legend()
plt.grid()
plt.show()

# u''(x) for the finest grid
u_xx = np.zeros_like(u)
h = x[1] - x[0]
u_xx[1:-1] = (u[0:-2] - 2*u[1:-1] + u[2:]) / h**2
plt.plot(x, u_xx, label="u''(x) from FD2")
plt.xlabel('x')
plt.ylabel('u\'\'(x)')
plt.title("Second derivative u''(x)")
plt.legend()
plt.grid()
plt.show()

# Plot error vs grid size
plt.figure()
plt.loglog(Ns[1:], errors, 'o-', label='Refinement error (max norm)')
plt.xlabel('Number of Points')
plt.ylabel('Error')
plt.title('Grid refinement error (no reference)')
plt.legend()
plt.grid()
plt.show()
print(f"Smallest observed grid refinement error: {min(errors):.2e} (max norm for N={Ns[-1]})")
