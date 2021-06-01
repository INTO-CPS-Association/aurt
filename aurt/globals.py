import math
import numpy as np
from aurt.num_sym_layers import npvector
import sympy as sp

Njoints = 6
MAXJOINTS = 6

G = np.array([
    [0.13408],
    [-0.07661],
    [-9.38527]
])
G_uni = G / np.linalg.norm(G)
g = G_uni * 9.81

g = npvector([0, 0, 9.81])
zero_gravity = npvector([0.0, 0.0, 0.0])

# Taken from robot_calibration_summary.txt
ur5e_Ktau = npvector([
    0.0,
    0.1005,
    0.1005,
    0.1003,
    0.0802,
    0.0812,
    0.0800,
])

# Taken from robot_calibration_summary.txt
vel_calibration = np.array(
    [-3.20,  -1.50,  -0.50,  -0.01,  +0.00,  +0.01,  +0.50,  +1.50,  +3.20])
I_calibration_joints_ur5e = np.array([
    [000000, 000000, 000000, 000000, 000000, 000000, 000000, 000000, 000000],  # Base joint
    [-1.805, -1.379, -0.934, -0.586, +0.000, +0.552, +0.944, +1.350, +1.732],  # Joint 1
    [-2.180, -1.570, -1.053, -0.652, +0.000, +0.634, +1.063, +1.595, +2.088],  # Joint 2
    [-1.790, -1.351, -0.884, -0.497, +0.000, +0.495, +0.891, +1.357, +1.822],  # Joint 3
    [-0.551, -0.420, -0.302, -0.175, +0.000, +0.175, +0.292, +0.407, +0.539],  # Joint 4
    [-0.464, -0.370, -0.253, -0.149, +0.000, +0.152, +0.247, +0.352, +0.458],  # Joint 5
    [-0.499, -0.383, -0.269, -0.156, +0.000, +0.157, +0.275, +0.356, +0.491],  # Joint 6
])


def get_ur3e_parameters(zeros_array):
    m = [
        None,
        1.90,
        3.4445,
        1.437,
        0.871,
        0.805,
        0.261
    ]

    d = zeros_array(MAXJOINTS + 1)
    a = zeros_array(MAXJOINTS + 1)
    alpha = zeros_array(MAXJOINTS + 1)

    # From https://www.universal-robots.com/articles/ur-articles/parameters-for-calculations-of-kinematics-and-dynamics/
    d[1] = 0.15185
    d[4] = 0.13105
    d[5] = 0.08535
    d[6] = 0.0921

    a[2] = -0.24355
    a[3] = -0.2132

    alpha[1] = math.pi / 2
    alpha[4] = math.pi / 2
    alpha[5] = -math.pi / 2

    return m, d, a, alpha


def get_ur5e_parameters(zeros_array):
    m = [
        None,
        3.761,
        8.058,
        2.846,
        1.37,
        1.3,
        0.365
    ]

    d = zeros_array(MAXJOINTS + 1)
    a = zeros_array(MAXJOINTS + 1)
    alpha = zeros_array(MAXJOINTS + 1)

    # From https://www.universal-robots.com/articles/ur-articles/parameters-for-calculations-of-kinematics-and-dynamics/
    d[1] = 0.1625
    d[4] = 0.1333
    d[5] = 0.0997
    d[6] = 0.0996

    a[2] = -0.425
    a[3] = -0.3922

    alpha[1] = math.pi / 2
    alpha[4] = math.pi / 2
    alpha[5] = -math.pi / 2

    return m, d, a, alpha


def get_ur_parameters_symbolic(zeros_array):
    m = [0] * (MAXJOINTS + 1)
    for j in range(1, MAXJOINTS + 1):
        m[j] = sp.symbols(f"m{j}")

    d = zeros_array(MAXJOINTS + 1)
    a = zeros_array(MAXJOINTS + 1)
    alpha = zeros_array(MAXJOINTS + 1)

    # From https://www.universal-robots.com/articles/ur-articles/parameters-for-calculations-of-kinematics-and-dynamics/
    d[1] = sp.symbols(f"d1")
    d[4] = sp.symbols(f"d4")
    d[5] = sp.symbols(f"d5")
    d[6] = sp.symbols(f"d6")

    a[2] = sp.symbols(f"a2")
    a[3] = sp.symbols(f"a3")

    alpha[1] = sp.pi / 2
    alpha[4] = sp.pi / 2
    alpha[5] = -sp.pi / 2

    return m, d, a, alpha


def get_P(a, d, alpha, vector, cos, sin):
    """
    P[i] means the position of frame i wrt to i-1
    P[2] = [d, -0.2]^T
    P[5] = [a[5-1], 0, d[5]] means the position of frame 5 wrt to 4
    """
    P = [vector([0, 0, 0]) for i in range(0, MAXJOINTS + 2)]

    for i in range(1, Njoints + 1):
        P[i] = vector([a[i - 1], -sin(alpha[i - 1]) * d[i], cos(alpha[i - 1]) * d[i]])

    return P


def get_ur_frames(a, vector):
    # PC[1] means the position of center of mass of link 1 wrt frame 1.
    PC = [vector([0.0, 0.0, 0.0])] * (MAXJOINTS + 1)
    for j in range(1, MAXJOINTS + 1):
        PC[j] = vector([sp.symbols(f"PCx{j}"), sp.symbols(f"PCy{j}"), sp.symbols(f"PCz{j}")])

    return PC


def get_ur3e_PC(a, vector):
    # PC[1] means the position of center of mass of link 1 wrt frame 1.
    # And link 1 connects frames/joints 1 to 2.
    """
    The parameters in https://www.universal-robots.com/articles/ur/parameters-for-calculations-of-kinematics-and-dynamics/
    Obey a different convention: Link i connects joints i to i+1, but its coordinates are specified in terms of joint i+1.
    """
    PC = [vector([0, 0, 0]) for i in range(Njoints + 1)]

    # PC.append(vector([0, -0.02, 0]))
    PC[1] = (vector([0, 0, -0.02]))

    # PC[2] = (vector([0.13, 0, 0.1157]))
    PC[2] = (vector([-(abs(a[2]) - 0.13), 0, 0.1157]))

    # PC[3] = (vector([0.05, 0, 0.0238]))
    PC[3] = (vector([-(abs(a[3]) - 0.05), 0, 0.0238]))

    # PC[4] = (vector([0, 0, 0.01]))
    PC[4] = (vector([0, -0.01, 0]))

    # PC[5] = (vector([0, 0, 0.01]))
    PC[5] = (vector([0, 0.01, 0]))

    # PC[6] = (vector([0, 0, -0.02]))
    PC[6] = (vector([0, 0, -0.02]))

    return PC


def get_ur5e_PC(a, vector):
    # PC[1] means the position of center of mass of link 1 wrt frame 1.

    PC = [vector([0, 0, 0]) for i in range(Njoints + 1)]

    # PC[1] = (vector([0, -0.02561, 0.00193]))
    PC[1] = (vector([0, -0.00193, -0.02561]))

    # PC[2] = (vector([0.2125, 0, 0.11336]))
    PC[2] = (vector([-(abs(a[2]) - 0.2125), 0, 0.11336]))

    # PC[3] = (vector([0.15, 0, 0.0265]))
    PC[3] = (vector([-(abs(a[3]) - 0.15), 0, 0.0265]))

    # PC[4] = (vector([0, -0.0018, 0.01634]))
    PC[4] = (vector([0, -0.01634, -0.0018]))

    # PC[5] = (vector([0, 0.0018, 0.01634]))
    PC[5] = (vector([0, 0.01634, -0.0018]))

    # PC[6] = (vector([0, 0, -0.02]))
    PC[6] = (vector([0, 0, -0.001159]))

    return PC
