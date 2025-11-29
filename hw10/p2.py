#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def exact_solution(x, t):
    """
    Exact solution for the diffusion equation
    Using separation of variables, the exact solution can be derived.
    For this specific problem with the given initial and boundary conditions,
    we can use Fourier series representation.
    
    u(x,t) = sum of Fourier modes that decay exponentially in time
    
    For the initial condition u(x,0) = sin(pi*x/2) + 0.5*sin(2*pi*x):
    - First term: sin(pi*x/2) decays as exp(-(pi/2)^2 * t)
    - Second term: 0.5*sin(2*pi*x) decays as exp(-(2*pi)^2 * t)
    
    The boundary condition u(1,t) = exp(-pi^2*t/4) is consistent with the first mode.
    """
    u_exact = (np.sin(np.pi * x / 2) * np.exp(-np.pi**2 * t / 4) + 
               0.5 * np.sin(2 * np.pi * x) * np.exp(-4 * np.pi**2 * t))
    return u_exact

def compute_errors(x, u_numerical, t):
    """
    Compute maximum error and L2 error
    
    Parameters:
    -----------
    x : array
        Spatial grid points
    u_numerical : array
        Numerical solution
    t : float
        Time at which to compute error
    
    Returns:
    --------
    max_error : float
        Maximum absolute error
    l2_error : float
        L2 norm of error
    """
    u_exact = exact_solution(x, t)
    error = np.abs(u_numerical - u_exact)
    
    # Maximum error (infinity norm)
    max_error = np.max(error)
    
    # L2 error
    dx = x[1] - x[0]
    l2_error = np.sqrt(np.sum(error**2) * dx)
    
    return max_error, l2_error, u_exact

def solve_diffusion_forward_euler(mu, T_final=0.7):
    """
    Solve the diffusion equation u_t = u_xx using Forward Euler method
    
    Equation: u_t = u_xx, 0 <= x <= 1, t >= 0
    Initial condition: u(x,0) = sin(pi*x/2) + 0.5*sin(2*pi*x)
    Boundary conditions: u(0,t) = 0, u(1,t) = exp(-pi^2*t/4)
    
    Parameters:
    -----------
    mu : float
        Stability parameter (mu = dt/dx^2)
    T_final : float
        Final time for simulation
    """
    
    # Spatial domain
    L = 1.0
    N = 100  # number of spatial intervals
    dx = L / N
    x = np.linspace(0, L, N+1)
    
    # Time step based on stability parameter
    dt = mu * dx**2
    M = int(T_final / dt)
    
    print(f"Simulation Parameters:")
    print(f"  mu = {mu}")
    print(f"  dx = {dx:.6f}")
    print(f"  dt = {dt:.6f}")
    print(f"  Number of time steps = {M}")
    print(f"  Stability condition: mu <= 0.5, current mu = {mu:.3f}")
    print()
    
    # Initialize solution array
    u = np.zeros(N+1)
    
    # Initial condition: u(x, 0) = sin(pi*x/2) + 0.5*sin(2*pi*x)
    u = np.sin(np.pi * x / 2) + 0.5 * np.sin(2 * np.pi * x)
    
    # Store solutions at different times for plotting
    times_to_save = [0, 0.2, 0.5, 0.7]
    saved_solutions = {}
    saved_errors = {}
    saved_solutions[0] = u.copy()
    
    # Compute initial error
    max_err, l2_err, _ = compute_errors(x, u, 0)
    saved_errors[0] = {'max_error': max_err, 'l2_error': l2_err}
    
    # Determine which time steps to save
    save_indices = []
    for t_save in times_to_save[1:]:  # Skip t=0 as it's already saved
        n_save = int(np.round(t_save / dt))
        save_indices.append(n_save)
    
    # Time stepping
    for n in range(M):
        u_new = np.zeros(N+1)
        
        # Boundary conditions
        u_new[0] = 0  # u(0, t) = 0
        t_current = (n + 1) * dt
        u_new[N] = np.exp(-np.pi**2 * t_current / 4)  # u(1, t) = exp(-pi^2*t/4)
        
        # Interior points using Forward Euler finite difference method
        # Discretization: u_t = u_xx
        # Time derivative (Forward Euler): du/dt ≈ (u_i^{n+1} - u_i^n) / dt
        # Space derivative (Central difference): d²u/dx² ≈ (u_{i+1}^n - 2*u_i^n + u_{i-1}^n) / dx²
        # Combined: (u_i^{n+1} - u_i^n) / dt = (u_{i+1}^n - 2*u_i^n + u_{i-1}^n) / dx²
        # Rearranging: u_i^{n+1} = u_i^n + (dt/dx²)*(u_{i+1}^n - 2*u_i^n + u_{i-1}^n)
        # Where μ = dt/dx² is the stability parameter
        for i in range(1, N):
            u_new[i] = u[i] + mu * (u[i+1] - 2*u[i] + u[i-1])
        
        u = u_new
        
        # Save solutions at specific time steps
        if (n + 1) in save_indices:
            actual_time = (n + 1) * dt
            # Find the closest requested time
            closest_t = min(times_to_save[1:], key=lambda t: abs(t - actual_time))
            if closest_t not in saved_solutions:
                saved_solutions[closest_t] = u.copy()
                # Compute errors
                max_err, l2_err, _ = compute_errors(x, u, actual_time)
                saved_errors[closest_t] = {'max_error': max_err, 'l2_error': l2_err}
                print(f"  Saved solution at t = {actual_time:.4f} (requested t = {closest_t})")
    
    # Make sure we have the final time if it's close to T_final
    if abs(M * dt - T_final) < dt and T_final in times_to_save:
        if T_final not in saved_solutions:
            saved_solutions[T_final] = u.copy()
            max_err, l2_err, _ = compute_errors(x, u, M * dt)
            saved_errors[T_final] = {'max_error': max_err, 'l2_error': l2_err}
            print(f"  Saved final solution at t = {M * dt:.4f}")
    
    # Print max values to check for instability
    print(f"\nMax absolute values at each saved time:")
    for t in sorted(saved_solutions.keys()):
        max_val = np.max(np.abs(saved_solutions[t]))
        print(f"  t = {t:.2f}: max|u| = {max_val:.6e}")
        if max_val > 100:
            print(f"    WARNING: Solution is UNSTABLE!")
    
    # Print errors
    print(f"\nErrors compared to exact solution:")
    print(f"{'Time':<10} {'Max Error':<15} {'L2 Error':<15}")
    print("-" * 40)
    for t in sorted(saved_errors.keys()):
        max_err = saved_errors[t]['max_error']
        l2_err = saved_errors[t]['l2_error']
        print(f"{t:<10.2f} {max_err:<15.6e} {l2_err:<15.6e}")
    
    return x, saved_solutions, saved_errors

def plot_solutions(x, solutions, errors=None, mu=None):
    """Plot the solutions at different times"""
    plt.figure(figsize=(12, 8))
    
    # Plot solutions at different times
    colors = plt.cm.viridis(np.linspace(0, 1, len(solutions)))
    
    # Check if solution has blown up (instability indicator)
    max_value = max([np.max(np.abs(u)) for u in solutions.values()])
    is_unstable = max_value > 10  # If values exceed 10, it's clearly unstable
    
    for (t, u), color in zip(sorted(solutions.items()), colors):
        # Plot numerical solution
        plt.plot(x, u, label=f't = {t:.2f} (max={np.max(np.abs(u)):.2e})', linewidth=2, color=color)
        
        # Plot exact solution
        u_exact = exact_solution(x, t)
        plt.plot(x, u_exact, '--', linewidth=1.5, color=color, alpha=0.6)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x, t)', fontsize=12)
    title = 'Solution: Numerical (solid) vs Exact (dashed)'
    if mu is not None:
        title += f' (μ = {mu})'
    plt.title(title, fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    
    if is_unstable:
        plt.text(0.5, 0.95, 'UNSTABLE - Solution Blowing Up!', 
                transform=plt.gca().transAxes,
                fontsize=14, color='red', fontweight='bold',
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('diffusion_solution.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_stability(T_final=0.7):
    """Compare solutions with different mu values"""
    plt.figure(figsize=(14, 5))
    
    mu_values = [0.5, 0.509]
    
    for idx, mu in enumerate(mu_values, 1):
        plt.subplot(1, 2, idx)
        x, solutions, errors = solve_diffusion_forward_euler(mu, T_final)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(solutions)))
        for (t, u), color in zip(sorted(solutions.items()), colors):
            u_exact = exact_solution(x, t)
            plt.plot(x, u, linewidth=2, color=color, label=f't = {t:.2f}')
            plt.plot(x, u_exact, '--', linewidth=1.5, color=color, alpha=0.6)
        
        plt.xlabel('x', fontsize=12)
        plt.ylabel('u(x, t)', fontsize=12)
        plt.title(f'μ = {mu} - Numerical (solid) vs Exact (dashed)', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        
        if mu > 0.5:
            plt.text(0.5, 0.95, 'UNSTABLE!', 
                    transform=plt.gca().transAxes,
                    fontsize=14, color='red', fontweight='bold',
                    ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('stability_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("Diffusion Equation Solver - Forward Euler Method")
    print("="*60)
    print()
    
    # Solve with mu = 0.5 (stable)
    print("CASE 1: mu = 0.5 (Stable)")
    print("-" * 60)
    x, solutions, errors = solve_diffusion_forward_euler(mu=0.5, T_final=0.7)
    plot_solutions(x, solutions, errors, mu=0.5)
    
    print("\n" + "="*60)
    print("CASE 2: mu = 0.509 (Potentially Unstable)")
    print("-" * 60)
    x, solutions, errors = solve_diffusion_forward_euler(mu=0.509, T_final=0.7)
    plot_solutions(x, solutions, errors, mu=0.509)
    
    # Compare both cases
    print("\n" + "="*60)
    print("Stability Comparison")
    print("="*60)
    compare_stability(T_final=0.7)