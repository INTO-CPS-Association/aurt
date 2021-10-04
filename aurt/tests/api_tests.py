import unittest
import pickle
from pathlib import Path
import sympy as sp
from aurt.file_system import from_cache, from_project_root
from aurt import api
from aurt.rigid_body_dynamics import RigidBodyDynamics
from aurt.tests.units import init_cache_dir


class APITests(unittest.TestCase):

    # The naming of the test cases control the order of the tests
    def test01_compile_rbd(self):
        init_cache_dir()
        mdh_path = str(from_project_root("aurt/tests/resources/twolink_dh.csv"))
        output_path = "rbd_twolink"
        plotting = False

        api.compile_rbd(mdh_path, output_path, plotting)
        filename = from_cache(output_path + ".pickle")
        with open(filename, 'rb') as f:
            rbd_twolink_estimate: RigidBodyDynamics = pickle.load(f)
        with open(str(from_project_root(Path("aurt/tests/resources", output_path))) + ".pickle", 'rb') as f:
            rbd_twolink_true: RigidBodyDynamics = pickle.load(f)

        self.assertEqual(rbd_twolink_estimate.mdh, rbd_twolink_true.mdh)
        self.assertEqual(rbd_twolink_estimate.n_params, rbd_twolink_true.n_params)
        self.assertEqual(rbd_twolink_estimate.params, rbd_twolink_true.params)
        tau_rbd = rbd_twolink_estimate.dynamics()
        self.assertEqual(tau_rbd, rbd_twolink_true.dynamics())

        M, C, g = rbd_twolink_estimate.euler_lagrange()
        qd = sp.Matrix(rbd_twolink_estimate.qd[1:])
        qdd = sp.Matrix(rbd_twolink_estimate.qdd[1:])

        self.assertEqual(sp.simplify(M*qdd + C*qd + g - tau_rbd), sp.zeros(rbd_twolink_estimate.n_joints, 1))

    def test02_compile_rbd_save_class(self):
        # test that class is saved properly
        output_path = "rbd_twolink"
        filename = from_cache(output_path + ".pickle")

        with open(filename, 'rb') as f:
            newrbd: RigidBodyDynamics = pickle.load(f)

        self.assertIsNotNone(newrbd.params, "The parameters are not set.")

    def test03_compile_rd(self):
        model_rbd = "rbd_twolink"
        friction_torque_model = "square"
        friction_viscous_powers = [2, 1, 4]
        output_file = "rd_twolink"

        api.compile_rd(model_rbd, friction_torque_model, friction_viscous_powers, output_file)

        filename = from_cache(output_file + ".pickle")
        with open(filename, 'rb') as f:
            rd_twolink_estimate = pickle.load(f)
        with open(str(from_project_root(Path("aurt/tests/resources", output_file))) + ".pickle", 'rb') as f:
            rd_twolink_true = pickle.load(f)

        self.assertEqual(rd_twolink_estimate.n_joints, rd_twolink_true.n_joints)
        self.assertEqual(rd_twolink_estimate.qdd, rd_twolink_true.qdd)
        self.assertEqual(rd_twolink_estimate.tauJ, rd_twolink_true.tauJ)

    def test04_calibrate(self):
        model_rd = "rd_twolink"
        data_file = str(from_project_root("aurt/tests/resources/twolink_data.csv"))
        gravity = [0, -9.81, 0]
        params_out = "twolink_params.csv"
        calibration_out = "rc_twolink"
        plotting = False

        api.calibrate(model_rd, data_file, gravity, params_out, calibration_out, plotting)

    def test05_predict(self):
        model = "rc_twolink"
        data = str(from_project_root("aurt/tests/resources/twolink_data.csv"))
        gravity = [0, -9.81, 0]
        prediction = "out_predict.csv"

        api.predict(model, data, gravity, prediction)

    def test06_plot_kinematics(self):
        mdh_path = str(from_project_root("aurt/tests/resources/twolink_dh.csv"))
        output_path = "rbd_twolink"
        plotting = True

        api.compile_rbd(mdh_path, output_path, plotting, block=False)


if __name__ == '__main__':
    unittest.main()
