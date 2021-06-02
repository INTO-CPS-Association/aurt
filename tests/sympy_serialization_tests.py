import os
import unittest

import sympy as sp

from aurt.file_system import store_object, load_object, cache_object
from tests.utils.timed_test import TimedTest
from tests import logger


class SerializationTests(TimedTest):

    def test_persistence_sympy(self):
        """
        Test that when we create a matrix and save it using sympy, we can load the same matrix again.
        """
        ftx, fty, ftz = sp.symbols("ftx fty ftz")
        ntx, nty, ntz = sp.symbols("ntx nty ntz")
        ft = sp.Matrix([ftx, fty, ftz])  # Load at the tip
        nt = sp.Matrix([ntx, nty, ntz])  # Moment at the tip

        matrix = ft * nt.transpose()

        logger.debug(matrix)

        store_object(matrix, 'matrix.txt')

        matrix2 = load_object('matrix.txt')

        logger.debug(matrix2)

        res = matrix2 - matrix

        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                self.assertAlmostEqual(0, sp.N(res[i,j]))

        os.remove("matrix.txt")

    def test_caching_sympy_expression(self):
        ftx, fty, ftz = sp.symbols("ftx fty ftz")
        ntx, nty, ntz = sp.symbols("ntx nty ntz")
        ft = sp.Matrix([ftx, fty, ftz])  # Load at the tip
        nt = sp.Matrix([ntx, nty, ntz])  # Moment at the tip

        matrix = ft * nt.transpose()

        cache_object('matrix', lambda: matrix)
        matrix2 = cache_object('matrix', lambda: matrix)

        res = matrix2 - matrix

        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                self.assertAlmostEqual(0, sp.N(res[i, j]))

        self.assertTrue(os.path.exists("matrix.pickle"))
        os.remove("matrix.pickle")



if __name__ == '__main__':
    unittest.main()
