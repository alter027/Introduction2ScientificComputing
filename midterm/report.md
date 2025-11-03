# Midterm Project
[TOC]

### The Main Problem
Before the AAA algorithm, rational approximation methods relied heavily on fixed support points or classical Padé approximations, often suffering from numerical instabilities like Froissart doublets. Existing algorithms lacked adaptability and robustness, especially on complex domains or scattered data, limiting accuracy and computational efficiency in practical applications.

#### Froissart doublets
When poles and zeros occur almost at the same location, their effect largely cancels, but the function's behaviour near that point can become numerically unstable or misleading.
It often results from numerical issues or overfitting, which can lead to misleading conclusions if not properly managed.

#### An Example
Consider $e^x$. A bad approximation of it would be

$$\frac{
    \left(1 + \frac{x}{2} + \frac{x^2}{12}\right) \left(1 - \frac{x}{5.1}\right) \left(1 + \frac{x}{5.1}\right)
}{
    \left(1 - \frac{x}{2} + \frac{x^2}{12}\right) \left(1 - \frac{x}{5}\right) \left(1 + \frac{x}{5}\right)
}$$

Then, poles appear at $5$ and $-5$, where zero appears near $5.1$ and $-5.1$, then we would find pole at $5$ is too close to zero at $5.1$, which behave far from the original function.
In comparison, following would be a better approximation since there isn't real closed poles and zeros.

$$\frac{1 + \frac{x}{2} + \frac{x^2}{12}}{1 - \frac{x}{2} + \frac{x^2}{12}}
$$

### Method in the Paper
#### Rational barycentric representations

$$r(z) = \frac{n(z)}{d(z)} = 
\frac{
    \displaystyle \sum_{j=1}^{m} \frac{w_j f_j}{z - z_j}
}{
    \displaystyle \sum_{j=1}^{m} \frac{w_j}{z - z_j}
}
$$

where $m \geq 1$ is an integer, $z_1, ... , z_m$ are a set of real of complext distinct support points, $f_1, ... , f_m$ are a set of real of complext data values, $w_1, ... , w_m$ are a set of real of complext weights.

#### Greedy Algorithm
Given a set of $z_1, ... , z_M$ and corresponding $f_1, ... , f_M$ for $\Omega = \{1, 2, ..., M\}$, $M$ is the size of the set. The keypoint of the algorithm is to find a set $Z  \subset \Omega$ and $Z' = \Omega\setminus S$.
In the beginning, $Z$ is an empty set. Choose an arbitrary point (?) and add it into $Z$ (Let's call it $Z_1$ now), then try to solve following Loewner matrix:

$$
A^{(m)} = 
\begin{pmatrix}
\displaystyle\frac{F^{(m)}_{1} - f^{*}_{1}}{Z^{(m)}_{1} - z_{1}} & \displaystyle\frac{F^{(m)}_{1} - f^{*}_{2}}{Z^{(m)}_{1} - z_{2}} & \cdots & \displaystyle\frac{F^{(m)}_{1} - f^{*}_{m}}{Z^{(m)}_{1} - z_{m}} \\
\vdots & \ddots & & \vdots \\
\displaystyle\frac{F^{(m)}_{M-m} - f^{*}_{1}}{Z^{(m)}_{M-m} - z_{1}} & \cdots & & \displaystyle\frac{F^{(m)}_{M-m} - f^{*}_{m}}{Z^{(m)}_{M-m} - z_{m}}
\end{pmatrix}
$$

where $m$ is size of $S$, $F^{(m)}=\{f_k\}$ where $k \in Z_m'$, and $f^{*} ={f_j}$ where $j \in Z_m$. In the first iteration, $m=1$, then we know that shape of $A^{(1)}$ is $(M-1, 1)$
Then, seek for $w$ where

$$
\text{minimize} \; \|A^{(m)} w \|_{M-m}, \quad \|w\|_m = 1,
$$

by using SVD.
After $w$ is solved, compute 

$$r^m(z) = \frac{n(z)}{d(z)} = 
\frac{
    \displaystyle \sum_{j=1}^{m} \frac{w_j f_j}{z - z_j}
}{
    \displaystyle \sum_{j=1}^{m} \frac{w_j}{z - z_j}
}
$$

and find the new supporting point $z_i$ from $Zm'$ by choosing

$$
arg\, max \,|\, r^m(z_i)-f(z_i) \,|
$$

 Then let $Z_{m+1}=Z_m \cup \{z_i\}$ and start the next iteration until all $|\, r^m(z_i)-f(z_i) \,|$ are less than the expected criterion.
### More Details
- Assume that $Z_m$ has at least $m$ points, i.e., $m \leq M/2$. It is to make the solution of $\|A^{(m)} w \|_{M-m}$ make sense.
- In the paper, it mentions the above method is usually good enough. However, if the numerical Froissart doublets still happen, for instance, they set the convergence tolerance to 0 and take the maximal step to sufficiently large. Their method is to identify spurious poles by setting residues < $10^{-13}$, remove the nearest point from the supporting point, and do SVD again.
- **Complexity**: Compared to other simple methods, AAA costs more to build the system, yet it is more effective on evaluation.

||**AAA**|**Barycentric**|**Cubic Spline**|
|---|---|---|---|
|Building System|$O(Mm^3)$|$O(M^2)$|$O(M)$|
|Evaluation|$O(m)$|$O(M)$|$O(M)$|

### Implementation
I used Python to implement the algorithm. Compilable code can be found in Github. Following is the pseudocode code for the main algorithm.

```=python3
def aaa(self):
    # initialization
    target_err = 1e-14

    # greedy
    for i in range(int(self.size/2)):
        self.s.append(_new_pt) # s is Z set in the Method

        # Cauchy Matrix
        s_c = np.delete(np.arange(self.size), self.s) # complement of s
        C = 1 / (self.x[s_c, None] - self.x[self.s][None, :])
        A = np.dot(SF, C) - np.dot(C, Sf)

        # Solve SVD
        _, _, Vh = np.linalg.svd(A, full_matrices = False)
        self.w = Vh[-1, :].conj()
        
        # Calculate r^m
        r = self.aaa_algorithm(self.x)

        # pick the new remaining points
        err = np.abs(self.y-r)

        _new_pt = np.argmax(np.abs(err))
        max_err = err[_new_pt]
        if max_err < target_err:
            break

    return self.aaa_algorithm
```

### Experiment Result and Observation
In my experiment, I compare **AAA algorithm** with **Barycentric** and **Cubic Spline** method.
Also, I use different sets of sample points which generated by **Chevyshev**, **equally space**, and **uniformly**.
#### 1. Experiment on cos(10x)
Consider $cos(10x)$ in $[\, -1, 1\,]$
||**uniformly**|**equally space**|**Chevyshev**|
|---|---|---|---|
|Maximal Error|<img src=_result/uniform_all_COS.jpg width="400"/>|<img src=_result/equal_space_all_COS.jpg width="400"/>|<img src=_result/chebyshev_all_COS.jpg width="400"/>|

- value of m
    - Using Chebyshev points
    - <img src=_result/chebyshev_cos.png width="400"/>
    - Using uniformly distribution
    - <img src=_result/uniform_cos.png width="400"/>



#### 2. Experiment on Gamma function (FIG 5.2)
Consider $cos(10x)$ in $[\, -1, 1\,]$
||**uniformly**|**equally space**|**Chevyshev**|
|---|---|---|---|
|Maximal Error|<img src=_result/uniform_all_FIG52.jpg width="400"/>|aaa fails on SVD, barycentric has a large err value, cubic_spline goes to infinite|<img src=_result/chebyshev_all_FIG52.jpg width="400"/>|

- value of m
    - Chevyshev
    - <img src=_result/chebyshev_fig.png width="400"/>
    - uniformly
    - <img src=_result/uniform_fig.png width="400"/>

#### Observation
- Barycentric seems to explode on uniform data, while Cubic Spline is hard to converge.
- With the greedy method, the AAA algorithm seems to avoid picking new closed support points from the set of support points and causing Froissart doublets.
- The AAA method works much better on uniformly random points compared with Barycentric or Cubic Spline interpolation method.
- the size of $Z$ for **AAA**, the value of $m$, usually less than $20$ in my experiments with sample sizes under $100$.

### Conclusion
- The AAA algorithm is mainly designed to concur with numerical Froissart doublets issue. The method includes barycentric representation, greedy selection on support points and solution with the SVD method.
- It provides a reliable interpolation method about stability and convergence on different distributions of data.
- About the complexity, even though the building time is more than other methods, there is an advantage on the evaluation. Also, the actual value of $m$ is usually small.