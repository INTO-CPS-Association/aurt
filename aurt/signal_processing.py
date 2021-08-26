import math
import numpy as np


def central_finite_difference(x, dt, order):
    """Computes the first and (optionally) second-order time-derivatives along axis 1. The dimension of the output is
    equal to the dimension of the input minus 2."""
    assert order in {1, 2}

    xd = np.gradient(x, dt, axis=1)

    if order == 2:
        xdd = (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]) / (dt**2)
        assert x[:, 1:-1].shape == xd[:, 1:-1].shape == xdd.shape
        return x[:, 1:-1], xd[:, 1:-1], xdd

    assert x[:, 1:-1].shape == xd[:, 1:-1].shape

    return x[:, 1:-1], xd[:, 1:-1]


def ramer_douglas_peucker(points, epsilon):
    def point_line_distance(point, start, end):
        if (start == end):
            def distance(a, b):
                return  math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
            
            return distance(point, start)
        else:
            n = abs((end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1]))
            d = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            return n / d
    
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d

    if dmax >= epsilon:
        results = ramer_douglas_peucker(points[:index+1], epsilon)[:-1] + ramer_douglas_peucker(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]

    return results
