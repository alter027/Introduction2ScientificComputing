# AAA algorithm equispaced

[TOC]

----
## Introduction
This paper proposes AAA rational approximation for interpolating smooth functions from equispaced data, demonstrating superior accuracy compared to existing methods. Through numerical experiments on various test functions, the authors show AAA typically converges twice as fast as alternatives and is particularly effective at exploiting analytic structure such as poles and branch points. The method balances the fundamental tradeoff between accuracy and stability in equispaced approximation, though it may encounter issues with unwanted poles. Theoretical analysis connects these results to the impossibility theorem for equispaced approximation.

### AAA approximation
AAA (Adaptive Antoulas-Anderson) approximation is a greedy algorithm that constructs rational function approximations (ratios of polynomials) to data. It iteratively selects support points and builds a barycentric representation, achieving high accuracy with relatively low degree. 
See more details in [midterm project](https://github.com/alter027/Introduction2ScientificComputing/blob/main/midterm/report.md).

### Impossibility theorem
The impossibility theorem for equispaced approximation states that exponential convergence and stability cannot be achieved simultaneously when approximating analytic functions from equispaced samples. Exponentially accurate approximations must be exponentially unstable as the number of sample points increases. Conversely, stable algorithms can only achieve root-exponential convergence at best, creating an unavoidable accuracy-stability tradeoff.

### An example
Consider polynomial least-squares approximation of the function $f(x) = \sqrt{1.21 - x²}$ from equispaced points [demonstration of impossibility theorem](https://claude.ai/public/artifacts/2d970bbe-a04a-4460-bd88-509327af46be):

- With well-conditioned basis (Chebyshev polynomials): 
$${1, x, x^2, x^3, ..., x^n}$$ The method achieves exponential accuracy initially, approximating the function to very high precision. However, it also becomes exponentially unstable-errors eventually grow at rate $\sim(1.14)^n$ due to amplification of rounding errors, causing the approximation to diverge catastrophically for large n.
- With ill-conditioned basis (monomials): 
$${T_0(x), T_1(x), T_2(x), ..., T_n(x)}$$ Rounding errors naturally regularize the problem, preventing both exponential accuracy and exponential instability. The method becomes stable but loses exponential convergence.

The theorem's core principle: you can have exponential accuracy with exponential instability, or stability with slower convergence, but not both simultaneously.

## 6 Methods
- **Cubic Splines**
    - **Convergence**: O(n⁻⁴) algebraic rate
    - **Stability**: Rock-solid stable
    - **Characteristics**: 
      - Piecewise polynomials with $C^2$ continuity
      - No Gibbs oscillations at boundaries
      - Slowest convergence but most predictable
      - Never unstable, never suffers from Runge phenomenon
- **Polynomial Least-Squares** (γ=2)
    - **Convergence**: Initially exponential, then diverges
    - **Stability**: Exponentially unstable O(1.14ⁿ) for γ=2
    - **Characteristics**:
      - Uses Chebyshev polynomial basis (well-conditioned)
      - Degree d ≈ n/2 (oversampling ratio γ=2)
      - Exponentially accurate AND exponentially unstable
      - Implementation-dependent: monomial basis cuts off both accuracy and instability

- **Fourier Extension to [-2,2]** (γ=2)
    - **Convergence**: Root-exponential
    - **Stability**: Stable (due to ill-conditioned basis)
    - **Characteristics**:
      - Approximates f by Fourier series on larger domain [-T,T]
      - Uses complex exponential basis (ill-conditioned)
      - Rounding errors provide implicit regularization
      - Implementation-dependent: Vandermonde-Arnoldi makes it exponentially accurate/unstable

- **Fourier + Low Degree Polynomial** (γ=2)
    - **Convergence**: Root-exponential
    - **Stability**: Stable
    - **Characteristics**:
      - Combines Fourier series with polynomial term of degree ~√n
      - Boundary corrections mitigate discontinuity effects
      - Algebraic convergence with arbitrary order possible
      - Neither exponentially accurate nor exponentially unstable

- **Floater-Hormann: 'equi'**
    - **Convergence**: Root-exponential (adaptively determined)
    - **Stability**: Stable
    - **Characteristics**:
      - Rational interpolation in barycentric form
      - Degree n-1 with adaptive accuracy order
      - Guaranteed pole-free in [-1,1]
      - Robust but doesn't exploit analytic structure
      - Matches AAA for "difficult" functions (e.g., amber function)

- **AAA**
    - **Convergence**: Typically 2x faster than other methods
    - **Stability**: Stable (barycentric representation)
    - **Characteristics**:
      - Nonlinear rational approximation
      - Exploits analytic structure (poles, branch points)
      - Numerical interpolant ($\leq 10^{-13}$ error on grid)
      - Can produce unwanted poles in [-1,1] for real functions
      - Best for functions with hidden analytic structure
      - Degree typically $<< n/2$ due to adaptive oversampling

### Complexity
| Method | Construction | Evaluation (single point) | Evaluation (m points) | Storage |
|--------|-------------|--------------------------|----------------------|---------|
| Cubic Splines | O(n) | O(log n) | O(m log n) | O(n) |
| Polynomial Least-Squares | O(nd²) | O(d) | O(md) | O(d) |
| Fourier Extension | O(nd) | O(d) | O(md) | O(d) |
| Fourier + Polynomial | O(nd) | O(d) | O(md) | O(d) |
| Floater-Hormann | O(n²d) | O(n) | O(mn) | O(n) |
| AAA | O(nd³) | O(d²) | O(md²) | O(d) |

where: n = number of sample points, d = degree of approximant, m = number of evaluation points

## Implementation in the paper

### Comparison betweem Methods
| Method | $\sqrt{1.21-x^2}$ | $\sqrt{0.01+x²}$ | $tanh(5x)$ | $sin(40x)$ | $exp(-1/x^2)$ |
|--------|-----------|-----------|----------|----------|------------|
| Cubic Splines | O(n⁻⁴) | O(n⁻⁴) | O(n⁻⁴) | O(n⁻⁴) | O(n⁻⁴) |
| Poly LS | Good→Unstable | Poor→Unstable | Good→Unstable | Good→Unstable | Poor→Unstable |
| Fourier Ext | Good | Good | Good | Good | Moderate |
| Fourier+Poly | Moderate | Moderate | Moderate | Moderate | Moderate |
| FH 'equi' | Very Good | Very Good | Good | Very Good | Very Good |
| **AAA** | **Best** | **Best** | **Exceptional** | **Best** | **Best** |

## Analysis
### Impossibility Theorem
- The impossibility theorem states that exponential accuracy in approximating analytic functions from equispaced samples is only possible if the algorithm also exhibits exponential instability.
- Conversely, any stable algorithm can achieve at best root-exponential convergence with the form $||f-r_n||= exp(-C\sqrt{n}$, where C > 0 is a constant.
- AAA circumvents this fundamental limitation through its use of adaptive oversampling and its nonlinear nature, which allows it to adjust the degree independently of the number of sample points.

### Why AAA Wins
1. **Exploits analytic structure**: AAA can identify and capture poles in meromorphic functions like tanh(5x), as well as efficiently approximate functions with branch point singularities like √(1.21-x²).
2. **Adaptive degree selection**: Unlike methods that use a degree tied to n, AAA adaptively selects a degree much lower than n/2, often finding that very low-degree rational functions suffice when analytic structure is present.
3. **Numerical interpolation philosophy**: By matching the data to a tolerance of 10⁻¹³ rather than interpolating exactly, AAA gains robustness against the exponential instabilities that plague exact high-degree polynomial interpolation.
4. **Barycentric form stability**: The use of barycentric representation for rational functions provides excellent numerical stability, avoiding many of the conditioning issues that arise with other representations.

## Result of the implementation from my side
Implemented by Python [here](https://github.com/alter027/Introduction2ScientificComputing/tree/main/final), the trend is consistent with the results of the paper.
|Function|Result|
|---|---|
|$\sqrt{1.21-x^2}$|<img src=_result/equal_space_all_FA.jpg width="400"/>|
|$\sqrt{0.01+x^2}$|<img src=_result/equal_space_all_FB.jpg width="400"/>|
|$tanh(5x)$|<img src=_result/equal_space_all_FC.jpg width="400"/>|
|$sin(40x)$|<img src=_result/equal_space_all_FD.jpg width="400"/>|
|$exp(-1/x^2)$|<img src=_result/equal_space_all_FE.jpg width="400"/>|

## Conclusion
AAA algorithm achieves near-exponential convergence on equispaced data by adaptively selecting support points in a greedy, nonlinear rational approximation process that exploits analytic structures like poles and branch points, often converging twice as fast as linear methods. It maintains stability through barycentric representation and a numerical interpolation tolerance (e.g., 10^{-13}), avoiding high-degree exact interpolation that triggers exponential instability as dictated by the impossibility theorem.