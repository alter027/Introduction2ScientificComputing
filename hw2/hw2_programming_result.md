# runge function $\frac{1}{1+25x^2}, x \in (-1, 1)$

picked 10000 equally spaced nodes and get the maximum error like,

```
def maximum_error(func_1, func_2):
    _x = np.linspace(-1, 1, 10000)
    _y1 = func_1(_x)
    _y2 = func_2(_x)
    return np.max(np.abs(_y1-_y2))
```

For equal_space + cubic_spline, the error is less than 1e-10 when there are N + 1 = 1580 nodes, N = 1579

For chebyshev + barycentric, the error is less than 1e-10 when there are N + 1 = 117 nodes, N=116

The overall picture is as follows,

<img src=_result/tri.jpg width="400" height="400" />