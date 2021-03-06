import logging
import unittest
import pickle
import sympy as sp

from aurt.caching import PersistentPickleCache
from aurt.file_system import from_cache, from_project_root
from aurt import api
from aurt.rigid_body_dynamics import RigidBodyDynamics
from aurt.robot_dynamics import RobotDynamics
from aurt.caching import clear_cache_dir

class APITests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Runs when class is loaded.
        """
        
        cls.cache_dir = from_project_root('cache')
        clear_cache_dir(cls.cache_dir)
        cls.cache = PersistentPickleCache(cls.cache_dir)
        logging.basicConfig(level=logging.WARNING)

    # The naming of the test cases control the order of the tests
    def test01_compile_rbd(self):
        mdh_path = str(from_project_root("aurt/tests/resources/twolink_dh.csv"))
        output_path = "rbd_twolink.pickle"
        plotting = False

        api.compile_rbd(mdh_path, output_path, plotting, self.cache)
        with open(output_path, 'rb') as f:
            rbd_twolink_estimate: RigidBodyDynamics = pickle.load(f)
        tau_rbd = rbd_twolink_estimate.dynamics()

        M, C, g = rbd_twolink_estimate.euler_lagrange()
        qd = sp.Matrix(rbd_twolink_estimate.qd[1:])
        qdd = sp.Matrix(rbd_twolink_estimate.qdd[1:])

        self.assertEqual(sp.simplify(M*qdd + C*qd + g - tau_rbd), sp.zeros(rbd_twolink_estimate.n_joints, 1))

    def test02_compile_rbd_save_class(self):
        # test that class is saved properly
        output_file = "rbd_twolink.pickle"
        output_path = output_file

        with open(output_path, 'rb') as f:
            newrbd: RigidBodyDynamics = pickle.load(f)

        self.assertIsNotNone(newrbd.parameters, "The parameters are not set.")

    def test03_compile_rd(self):
        model_rbd = "rbd_twolink.pickle"
        friction_torque_model = "square"
        friction_viscous_powers = [2, 1, 4]
        output_file = "rd_twolink.pickle"
        output_path = output_file

        api.compile_rd(model_rbd, friction_torque_model, friction_viscous_powers, output_path, self.cache)

        with open(output_path, 'rb') as f:
            rd: RobotDynamics = pickle.load(f)
        
        self.assertEqual(rd.n_joints, 2)

    def test04_calibrate(self):
        model_rd = "rd_twolink.pickle"
        data_file = str(from_project_root("aurt/tests/resources/twolink_data.csv"))
        gravity = [0, -9.81, 0]
        params_out = from_cache("twolink_params.csv")
        calibration_out = "rc_twolink.pickle"
        plotting = False

        api.calibrate(model_rd, data_file, gravity, params_out, calibration_out, self.cache, plotting)

    def test05_predict(self):
        model = "rc_twolink.pickle"
        data = str(from_project_root("aurt/tests/resources/twolink_data.csv"))
        gravity = [0, -9.81, 0]
        prediction = from_cache("out_predict.csv")

        api.predict(model, data, gravity, prediction, self.cache)

    def test06_calibrate_validate(self):
        model_rd = "rd_twolink.pickle"
        data_file = str(from_project_root("aurt/tests/resources/twolink_data.csv"))
        gravity = [0, -9.81, 0]
        params_out = from_cache("twolink_params.csv")
        calibration_out = "rc_twolink.pickle"
        prediction = from_cache("out_predict.csv")
        plotting = False
        calibration_data_relative = 0.8

        api.calibrate_validate(model_rd, data_file, gravity, calibration_data_relative,
                               params_out, calibration_out, prediction, self.cache, plotting)


if __name__ == '__main__':
    unittest.main()
