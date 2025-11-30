import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dst, idst

# Compute exact solution using Fourier series
def compute_fourier_coeffs(N_modes=100):
    x = np.linspace(0, 1, 5000)
    return np.array([2 * np.trapz(np.sin(2*np.pi*x) * np.exp(x) * np.sin(k*np.pi*x), x) 
                     for k in range(1, N_modes+1)])

FOURIER_COEFFS = compute_fourier_coeffs()

def exact_solution(x, t):
    u = np.zeros_like(x)
    for k, b_k in enumerate(FOURIER_COEFFS, 1):
        u += b_k * np.sin(k*np.pi*x) * np.exp(-k**2 * np.pi**2 * t)
    return u

def solve_backward_euler_fst(N, T, dt):
    """Part (a): Backward Euler + FST"""
    h = 1.0 / (N + 1)
    x = np.linspace(h, 1-h, N)
    u = np.sin(2*np.pi*x) * np.exp(x)
    k = np.arange(1, N+1)
    lam = -4.0/h**2 * np.sin(np.pi*k/(2*(N+1)))**2
    
    u_freq = dst(u, type=1)
    for _ in range(int(T / dt)):
        u_freq /= (1.0 - dt * lam)
    u = idst(u_freq, type=1) / (2 * (N + 1))
    
    return np.concatenate(([0], x, [1])), np.concatenate(([0], u, [0]))

def solve_forward_euler_fst(N, T):
    """Part (b): Forward Euler + FST with adaptive dt"""
    h = 1.0 / (N + 1)
    x = np.linspace(h, 1-h, N)
    u = np.sin(2*np.pi*x) * np.exp(x)
    k = np.arange(1, N+1)
    lam = -4.0/h**2 * np.sin(np.pi*k/(2*(N+1)))**2
    
    dt = -1.0 / np.max(lam)  # Stable dt
    num_steps = int(np.ceil(T / dt))
    dt = T / num_steps
    
    u_freq = dst(u, type=1)
    for _ in range(num_steps):
        u_freq *= (1.0 + dt * lam)
    u = idst(u_freq, type=1) / (2 * (N + 1))
    
    return np.concatenate(([0], x, [1])), np.concatenate(([0], u, [0]))

def solve_mol_exact_fst(N, T):
    """Part (c): Method of Lines + FST with exact time"""
    h = 1.0 / (N + 1)
    x = np.linspace(h, 1-h, N)
    u = np.sin(2*np.pi*x) * np.exp(x)
    k = np.arange(1, N+1)
    lam = -4.0/h**2 * np.sin(np.pi*k/(2*(N+1)))**2
    
    u_freq = dst(u, type=1)
    u_freq *= np.exp(lam * T)
    u = idst(u_freq, type=1) / (2 * (N + 1))
    
    return np.concatenate(([0], x, [1])), np.concatenate(([0], u, [0]))

def plot_solutions(sols):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ['Backward Euler + FST', 'Forward Euler + FST', 'Method of Lines + FST']
    labels = ['Part a', 'Part b', 'Part c']
    blue_labels = ['Backward Euler (FST)', 'Forward Euler (FST)', 'MOL exact time (FST)']
    
    for ax, title, lbl, blue_lbl in zip(axes, titles, labels, blue_labels):
        for name, (x, u) in sols.items():
            if lbl in name and not np.any(np.isnan(u)):
                ax.plot(x, u, '-', lw=2.5, color='blue', label=blue_lbl)
            elif name == 'Exact':
                ax.plot(x, u, '--', lw=2, label='Exact solution', color='orange')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, T)')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("Problem 3: Diffusion Equation u_t = u_xx")
    print("="*60)
    
    N, T, dt = 100, 1.0, 0.001
    
    x_a, u_a = solve_backward_euler_fst(N, T, dt)
    x_b, u_b = solve_forward_euler_fst(N, T)
    x_c, u_c = solve_mol_exact_fst(N, T)
    u_exact = exact_solution(x_a, T)
    
    sols = {
        'Part a': (x_a, u_a),
        'Part b': (x_b, u_b),
        'Part c': (x_c, u_c),
        'Exact': (x_a, u_exact)
    }
    
    plot_solutions(sols)
    
    # Convergence analysis
    print("\nCONVERGENCE ANALYSIS")
    print("-"*60)
    
    x_ref, u_ref = solve_mol_exact_fst(800, T)
    N_vals = [20, 40, 80, 160]
    results = {}
    
    for name, func in [('BE', lambda N: solve_backward_euler_fst(N, T, dt)),
                       ('FE', lambda N: solve_forward_euler_fst(N, T)),
                       ('MOL', lambda N: solve_mol_exact_fst(N, T))]:
        errs = []
        for N_val in N_vals:
            x, u = func(N_val)
            if not np.any(np.isnan(u)):
                u_int = np.interp(x, x_ref, u_ref)
                errs.append(np.sqrt(np.sum((u - u_int)**2) * (x[1]-x[0])))
        results[name] = errs
    
    print(f"{'N':<6} | {'MOL':<12} | {'BE':<12} | {'FE':<12}")
    print("-"*54)
    for i, N_val in enumerate(N_vals):
        row = f"{N_val:<6}"
        for name in ['MOL', 'BE', 'FE']:
            if i < len(results[name]):
                row += f" | {results[name][i]:<12.4e}"
            else:
                row += f" | {'---':<12}"
        print(row)
    
    print("-"*54)
    for name, errs in results.items():
        if len(errs) >= 2:
            order = np.log(errs[-2]/errs[-1]) / np.log(2)
            print(f"{name} Convergence Order: {order:.2f}")
    
    print("\n" + "="*60)