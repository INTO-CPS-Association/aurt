import numpy as np


def central_finite_difference(x, dt, order):
    """Computes the first and (optionally) second-order time-derivatives alon axis 1. The dimension of the output is
    equal to the dimension of the input minus 2."""
    assert order in {1, 2}

    xd = np.gradient(x, dt, axis=1)

    if order == 2:
        xdd = (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]) / (dt**2)
        assert x[:, 1:-1].shape == xd[:, 1:-1].shape == xdd.shape
        return x[:, 1:-1], xd[:, 1:-1], xdd

    assert x[:, 1:-1].shape == xd[:, 1:-1].shape

    return x[:, 1:-1], xd[:, 1:-1]


def crop(x, idx_start=None, idx_end=None):
    assert x.shape[1] > idx_end, f"'idx_end' must be smaller than the length of the signal along axis 1"

    if idx_end is not None:
        x = x[:, :idx_end]
    if idx_start is not None:
        x = x[:, idx_start:]

    return x
