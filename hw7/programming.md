## Numerical Solution of -u'' = exp(sin(x)) with Dirichlet Conditions

The problem was solved using a fourth-order finite difference method (FD4) with mesh sizes from 32 to 1024 points. The FD4 scheme uses a 5-point stencil for increased accuracy, reverting to second-order near boundaries.

Accuracy was verified via grid refinement, comparing solutions with interpolated previous results and estimating truncation error from scaled second differences.

Results show consistent convergence with error declining approximately as the fourth power of grid spacing. The smallest error (~4e-6) matches truncation error estimates, confirming reliable accuracy assessment.

## P1 Results
Smallest observed error by grid refinement: 4.3603e-04

Estimated truncation error at finest grid: 5.9451e-12

Finest grid used: 1024 points

<img src=_result/p1_result.png width="400"/>
<img src=_result/p1_error.png width="400"/>

## P2 Results
Smallest observed grid refinement error: 1.59e-06 (max norm for N=512)

<img src=_result/p2_result_u.png width="400"/>
<img src=_result/p2_result_u2.png width="400"/>
<img src=_result/p2_error.png width="400"/>