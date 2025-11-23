import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

def p0_f(x):
    """Default p(x) = 0"""
    return 0.0

def q0_f(x):
    """Default q(x) = 0"""
    return 0.0

def solve_bvp_fdm(f, a, b, N, ya=None, yb=None, d_ya=None, d_yb=None, 
                  p_f=p0_f, q_f=q0_f):
    """
    Solve boundary value problem using 2nd-order finite difference method.
    
    Solves: u''(x) + p(x)u'(x) + q(x)u(x) = f(x) on [a, b]
    
    Parameters:
    -----------
    f : function
        Right-hand side function f(x)
    a, b : float
        Domain boundaries [a, b]
    N : int
        Number of interior points
    ya, yb : float, optional
        Dirichlet boundary conditions u(a)=ya, u(b)=yb
    d_ya, d_yb : float, optional
        Neumann boundary conditions u'(a)=d_ya, u'(b)=d_yb
    p_f : function, optional
        Coefficient p(x) in u'' + p(x)u' + q(x)u = f(x)
    q_f : function, optional
        Coefficient q(x) in u'' + p(x)u' + q(x)u = f(x)
    
    Returns:
    --------
    x : ndarray
        Grid points including boundaries
    u : ndarray
        Solution values at grid points
    """
    
    # Grid setup
    h = (b - a) / (N + 1)
    x = np.linspace(a, b, N + 2)
    
    # Initialize system matrix A and right-hand side vector b_vec
    A = np.zeros((N + 2, N + 2))
    b_vec = np.zeros(N + 2)
    
    # Interior points: central difference for u'', u'
    for i in range(1, N + 1):
        xi = x[i]
        p_val = p_f(xi)
        q_val = q_f(xi)
        
        # u''(xi) ≈ (u[i-1] - 2u[i] + u[i+1]) / h^2
        # u'(xi) ≈ (u[i+1] - u[i-1]) / (2h)
        
        A[i, i-1] = 1/h**2 - p_val/(2*h)
        A[i, i] = -2/h**2 + q_val
        A[i, i+1] = 1/h**2 + p_val/(2*h)
        b_vec[i] = f(xi)
    
    # Boundary conditions at x = a (i = 0)
    if ya is not None:
        # Dirichlet: u(a) = ya
        A[0, 0] = 1
        b_vec[0] = ya
    elif d_ya is not None:
        # Neumann: u'(a) = d_ya
        # Forward difference: u'(a) ≈ (-3u[0] + 4u[1] - u[2]) / (2h)
        A[0, 0] = -3/(2*h)
        A[0, 1] = 4/(2*h)
        A[0, 2] = -1/(2*h)
        b_vec[0] = d_ya
    else:
        raise ValueError("Must specify either ya or d_ya for left boundary")
    
    # Boundary conditions at x = b (i = N+1)
    if yb is not None:
        # Dirichlet: u(b) = yb
        A[N+1, N+1] = 1
        b_vec[N+1] = yb
    elif d_yb is not None:
        # Neumann: u'(b) = d_yb
        # Backward difference: u'(b) ≈ (u[N-1] - 4u[N] + 3u[N+1]) / (2h)
        A[N+1, N-1] = 1/(2*h)
        A[N+1, N] = -4/(2*h)
        A[N+1, N+1] = 3/(2*h)
        b_vec[N+1] = d_yb
    else:
        raise ValueError("Must specify either yb or d_yb for right boundary")
    
    # Solve the linear system
    u = np.linalg.solve(A, b_vec)
    
    return x, u

def plot_bvp(x_sol, y_sol, y_exact, u_name, N):
    # Plotting the solution
    plt.figure(figsize=(8, 6))
    plt.plot(x_sol, y_sol, label='FDM Solution')
    plt.plot(x_sol, y_exact, 'k--', label='Exact Solution') # Exact solution for this example
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Finite Difference Method for '+u_name)
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('_result/'+u_name.split('_')[0]+'_result_sample'+str(N), dpi=96)
    plt.close()