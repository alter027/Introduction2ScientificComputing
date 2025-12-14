import numpy as np
from scipy.interpolate import barycentric_interpolate, CubicSpline 

class Interpolate:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = len(self.x)
    def method(self, method):
        self.func = getattr(self, method)()
        return self.func
    def cubic_spline(self):
        return CubicSpline(self.x, self.y)
    def barycentric(self):
        return lambda _x: barycentric_interpolate(self.x, self.y, _x)
    def aaa(self):
        # initial 
        target_err = 1e-14
        self.w = np.array([]) # weights
        self.s = [] # set of current supporting points
        r = np.ones(self.size) + np.mean(self.y) # output of remaining points
        _new_pt = np.argmax(np.abs(self.y-r))
        # greedy
        for i in range(int(self.size/2)):
            self.s.append(_new_pt)
            # Cauchy Matrix
            s_c = np.delete(np.arange(self.size), self.s)
            C = 1 / (self.x[s_c, None] - self.x[self.s][None, :])
            SF = np.diag(self.y[s_c])
            Sf = np.diag(self.y[self.s])
            A = np.dot(SF, C) - np.dot(C, Sf)
            # Solve SVD
            _, _, Vh = np.linalg.svd(A, full_matrices = False)
            self.w = Vh[-1, :].conj()
            r = self.aaa_algorithm(self.x)
            # pick the new remaining points
            err = np.abs(self.y-r)
            _new_pt = np.argmax(np.abs(err))
            max_err = err[_new_pt]
            if max_err < target_err:
                if (self.size % 10 == 0):
                    print("Sample size:", self.size, ", m:", len(self.s))
                break
        return self.aaa_algorithm
    def aaa_algorithm(self, _xs):
        ret = []
        # rational barycentric
        for _x in _xs:
            if _x in self.x[self.s]:
                _i = np.where(self.x == _x)
                ret.append(self.y[_i][0])
            else:
                num = np.sum((self.w*self.y[self.s])/(_x-self.x[self.s]))
                dom = np.sum(self.w/(_x-self.x[self.s]))
                ret.append(num/dom)
        return ret

    def poly_least_square(self, gamma=2):
        """
        Polynomial least-squares with oversampling ratio gamma.
        From paper: degree d ≈ n/gamma to cut exponential growth.
        """
        n = self.size
        d = n // gamma  # degree of polynomial
        
        # Use Chebyshev basis for well-conditioned computation
        # Generate Chebyshev polynomials up to degree d
        def chebyshev_vandermonde(x, degree):
            """Generate Chebyshev Vandermonde matrix"""
            T = np.zeros((len(x), degree + 1))
            T[:, 0] = 1
            if degree > 0:
                T[:, 1] = x
            for k in range(2, degree + 1):
                T[:, k] = 2 * x * T[:, k-1] - T[:, k-2]
            return T
        
        # Fit coefficients using least squares
        A = chebyshev_vandermonde(self.x, d)
        c = np.linalg.lstsq(A, self.y, rcond=None)[0]
        
        # Return interpolation function
        def interpolant(_x):
            A_eval = chebyshev_vandermonde(_x, d)
            return A_eval @ c
        
        return interpolant

    def fourier_extension(self, T=2, gamma=2):
        x = self.x
        y = self.y
        """Fourier extension on [-T, T] with oversampling ratio gamma."""
        n = len(x)
        d = int(np.ceil(n / (4 * gamma)))
        
        Z = np.exp(0.5j * np.pi * x)
        A = np.column_stack([np.real(Z ** k) for k in range(d + 1)] + 
                            [np.imag(Z ** k) for k in range(1, d + 1)])
        
        c = np.linalg.lstsq(A, y, rcond=None)[0]
        c_complex = c[:d+1] - 1j * np.concatenate([[0], c[d+1:2*d+1]])
        
        def interpolant(x_eval):
            x_eval = np.atleast_1d(x_eval)
            zz = np.exp(0.5j * np.pi * x_eval)
            powers = np.column_stack([zz ** k for k in range(len(c_complex))])
            return np.real(powers @ c_complex)
        
        return interpolant

    def fourier_series_poly_degree(self, gamma=2):
        """
        Fourier series plus polynomial of degree sqrt(n) with oversampling ratio gamma.
        From paper: combines Fourier and polynomial terms for boundary corrections.
        """
        n = self.size
        degp = int(np.round(np.sqrt(n))) - 1  # degree of polynomial term
        degp = degp + (degp + n + 1) % 2  # adjust parity to match n
        degf = (n - 1 - degp) // 2
        degf = int(np.ceil(degf / 2))  # oversampling ratio about 2
        
        # Build design matrix
        # Fourier terms: cos(pi*k*x) and sin(pi*k*x) for k=1 to degf
        A_cos = np.column_stack([np.cos(np.pi * k * self.x) for k in range(1, degf + 1)])
        A_sin = np.column_stack([np.sin(np.pi * k * self.x) for k in range(1, degf + 1)])
        
        # Chebyshev polynomial terms up to degree degp
        def chebyshev_vandermonde(x, degree):
            T = np.zeros((len(x), degree + 1))
            T[:, 0] = 1
            if degree > 0:
                T[:, 1] = x
            for k in range(2, degree + 1):
                T[:, k] = 2 * x * T[:, k-1] - T[:, k-2]
            return T
        
        A_poly = chebyshev_vandermonde(self.x, degp)
        
        # Combine all terms
        A = np.hstack([A_cos, A_sin, A_poly])
        
        # Least squares fit
        c = np.linalg.lstsq(A, self.y, rcond=None)[0]
        
        # Return interpolation function
        def interpolant(_x):
            A_cos_eval = np.column_stack([np.cos(np.pi * k * _x) for k in range(1, degf + 1)])
            A_sin_eval = np.column_stack([np.sin(np.pi * k * _x) for k in range(1, degf + 1)])
            A_poly_eval = chebyshev_vandermonde(_x, degp)
            A_eval = np.hstack([A_cos_eval, A_sin_eval, A_poly_eval])
            return A_eval @ c
        
        return interpolant
    def floater_hormann(self):
        x = self.x
        y = self.y
        """
        Floater-Hormann barycentric rational interpolation for equispaced nodes.
        """
        n = len(x)
        
        # Choose blending parameter d
        if n <= 10:
            d = n - 1
        elif n <= 40:
            d = min(n - 1, 10)
        elif n <= 100:
            d = 8
        elif n <= 200:
            d = 7
        else:
            d = 6
        
        # Compute barycentric weights
        w = np.zeros(n)
        for i in range(n):
            w_sum = 0.0
            for k in range(max(0, i - d), min(i, n - d - 1) + 1):
                prod = 1.0
                for j in range(k, min(k + d + 1, n)):
                    if j != i:
                        prod *= (x[i] - x[j])
                if abs(prod) > 1e-14:
                    w_sum += (-1) ** k / prod
            w[i] = w_sum
        
        # Normalize weights
        w_max = np.max(np.abs(w))
        if w_max > 0:
            w = w / w_max
        
        def interpolant(x_eval):
            x_eval = np.atleast_1d(x_eval)
            result = np.zeros(len(x_eval))
            
            for idx, xval in enumerate(x_eval):
                diff = x - xval
                if np.any(np.abs(diff) < 1e-12):
                    result[idx] = y[np.argmin(np.abs(diff))]
                else:
                    inv_diff = 1.0 / diff
                    result[idx] = np.sum(w * y * inv_diff) / np.sum(w * inv_diff)
            
            return result
        
        return interpolant