import math
import numpy as np
import sympy as sp


# Numerical computation layer

def npcross(a, b):
    assert a.shape == (3, 1), a
    assert b.shape == (3, 1), b
    res = np.cross(a, b, axisa=0, axisb=0).transpose()
    assert res.shape == (3, 1)
    return res


def npdot(a, b):
    (n1, m1) = a.shape
    (n2, m2) = b.shape
    res = np.dot(a, b)
    assert res.shape == (n1, m2), a.shape
    return res


def npvector(l): return np.array(l).transpose().reshape((len(l), 1))


def npmatrix(l): return np.array(l)


def npzeros_array(n1): return np.zeros(n1)


def npzeros_matrix(n1, n2): return np.zeros((n1, n2))


def npcos(n): return math.cos(n)


def npsin(n): return math.sin(n)


def npeye(n): return np.eye(n)

# Symbolic computation layer

def spcross(v1, v2):
    assert v1.shape == (3, 1), v1
    assert v2.shape == (3, 1), v2
    res = v1.transpose().cross(v2.transpose()).transpose()
    assert res.shape == (3, 1)
    return res


def spdot(a, b):
    (n1, m1) = a.shape
    (n2, m2) = b.shape
    res = a * b
    assert res.shape == (n1, m2)
    return res


def spvector(l): return sp.Matrix(l)


def spmatrix(l): return sp.Matrix(l)


def spzeros_array(n1): return sp.zeros(rows=n1, cols=1)


def spzeros_matrix(n1, n2): return sp.zeros(n1, n2)


def spcos(n): return sp.cos(n)


def spsin(n): return sp.sin(n)


def speye(n): return sp.eye(n)