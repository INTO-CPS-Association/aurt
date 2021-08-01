
import unittest
import pickle
import os
import sympy as sp


from aurt import api
from aurt.data_processing import ModifiedDH
from aurt.rigid_body_dynamics import RigidBodyDynamics

class RBDTests(unittest.TestCase):

    def test_rigid_body_dynamics(self):
        mdh = ModifiedDH([0,0,0], [0,0,1], [0,0,0], None)
        q = [sp.Integer(0)] + [sp.symbols(f"q{j}") for j in range(1, mdh.n_joints + 1)]
        mdh.q = q
        gravity = [0, -9.81, 0]
        output_path = "rbd_test"
        rbd = RigidBodyDynamics(mdh, gravity)
        rbd.regressor(output_path)

        filename = os.path.join(os.getcwd(),"cache", output_path + ".pickle")
        with open(filename, 'rb') as f:
            rbd_twolink_estimate = pickle.load(f)
        with open("tests/resources/rbd_twolink.pickle", 'rb') as f:
            rbd_twolink_true = pickle.load(f)

        self.assertEqual(rbd_twolink_estimate,rbd_twolink_true)
