import math
import os
import unittest

import numpy as np
import sympy as sp

from aurt.globals import Njoints, g, get_ur5e_parameters, get_ur5e_PC
from aurt.num_sym_layers import npcross, npvector, npzeros_array
from tests.utils.plotting import draw_robot
from aurt.torques import compute_torques_numeric_3e, compute_torques_symbolic_3e, compute_torques_numeric_5e
from aurt.file_system import cache_numpy, store_numpy_expr, load_numpy_expr
from tests import NONINTERACTIVE
import matplotlib.pyplot as plt

from tests.utils.timed_test import TimedTest
from tests import logger


class NumericalTests(TimedTest):

    def test_numerical_symbolic_same(self):

        # TODO: Need to run this test for multiple joint configurations

        # Numerical inputs
        q = np.zeros(Njoints + 1)
        der_q = np.zeros(Njoints + 1)
        der2_q = np.zeros(Njoints + 1)

        f_tip = np.zeros((3, 1))  # Load at the tip
        n_tip = np.zeros((3, 1))  # Moment at the tip

        # Dummy moments of inertia
        cI = [np.identity(3) for i in range(0, Njoints + 1)]

        tau_num = compute_torques_numeric_3e(q, der_q, der2_q, f_tip, n_tip, cI, g)

        # Symbolic inputs, as numbers
        q = [0.0 for i in range(Njoints + 1)]
        der_q = [0.0 for i in range(Njoints + 1)]
        der2_q = [0.0 for i in range(Njoints + 1)]

        ft = sp.Matrix([0, 0, 0])
        nt = sp.Matrix([0, 0, 0])

        tau_sym = compute_torques_symbolic_3e(q, der_q, der2_q, ft, nt, cI, g)

        assert len(tau_num) == len(tau_sym)

        for i in range(0, len(tau_num)):
            assert np.isclose(float(tau_sym[i]), tau_num[i])

        return True

    def test_numeric_torque(self):
        # Inputs
        q = np.zeros(Njoints + 1)
        der_q = np.zeros(Njoints + 1)
        der2_q = np.zeros(Njoints + 1)

        q[2] = 2*math.pi/2
        q[4] = -math.pi/2

        f_tip = np.zeros((3, 1))  # Load at the tip
        n_tip = np.zeros((3, 1))  # Moment at the tip

        cI = [np.zeros((3,3)) for j in range(Njoints + 1)]

        tau_num = compute_torques_numeric_5e(q, der_q, der2_q, f_tip, n_tip, cI, g)

        logger.debug(tau_num)

        qs = []
        q0 = np.zeros(Njoints + 1)
        qs.append(q0)
        qs.append(q)

        (_, d, a, alpha) = get_ur5e_parameters(npzeros_array)
        PC = get_ur5e_PC(a, npvector)
        ani = draw_robot(d, a, alpha, PC,
                         qs, 2000, repeat=True)

        if not NONINTERACTIVE:
            plt.show()
        plt.close()

    def test_save_load_numpy_array(self):
        arr = np.array([1, 2, 3])
        file = 'myNumpyArray.npy'

        def fun(a):
            for i in range(len(a)):
                a[i] += 2

        store_numpy_expr(arr, file)
        load_numpy_expr(file)
        cache_numpy(file, fun(arr))

        self.assertTrue(os.path.exists(file))
        os.remove(file)

    def test_cross_product(self):
        for a1 in np.linspace(0, 10, 3):
            for a2 in np.linspace(0, 10, 3):
                for a3 in np.linspace(0, 10, 3):
                    for b1 in np.linspace(0, 10, 3):
                        for b2 in np.linspace(0, 10, 3):
                            for b3 in np.linspace(0, 10, 3):
                                a = np.array([a1, a2, a3]).transpose().reshape((3, 1))
                                b = np.array([b1, b2, b3]).transpose().reshape((3, 1))
                                res = npcross(a, b)
                                self.assertAlmostEqual(res[0,0], a[1,0]*b[2,0] - a[2,0]*b[1,0])
                                self.assertAlmostEqual(res[1,0], a[2,0]*b[0,0] - a[0,0]*b[2,0])

   


if __name__ == '__main__':
    unittest.main()
