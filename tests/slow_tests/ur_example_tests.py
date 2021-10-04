import os.path
import pickle
import unittest

from aurt import api
from aurt.caching import PersistentPickleCache
from aurt.file_system import from_project_root, from_cache
from aurt.rigid_body_dynamics import RigidBodyDynamics
from aurt.tests.units import init_cache_dir


class URExampleTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Runs when class is loaded.
        """
        cls.cache_dir = from_project_root('cache')
        init_cache_dir(cls.cache_dir)
        cls.cache = PersistentPickleCache(cls.cache_dir)

    def test_ur5e_dh(self):
        mdh_path = str(from_project_root("resources/robot_parameters/ur5e_dh.csv"))
        gravity = [0.0, 6.937, -6.937]
        out_rbd = from_cache("rigid_body_dynamics.pickle")
        out_rd = from_cache("rd_twolink.pickle")
        friction_load_model = "square"
        friction_viscous_powers = [2, 1, 4]
        data_file = str(from_project_root("resources/Dataset/ur5e_45degX_aurt_demo_1/ur5e_45degX_aurt_demo_1.csv"))
        params_out = from_cache("ur5e_params.csv")
        calibration_out = from_cache("rc_ur5e.pickle")
        prediction = from_cache("out_predict.csv")

        # Compile RBD
        self.assertFalse(os.path.isfile(out_rbd))
        api.compile_rbd(mdh_path, out_rbd, False, self.cache)
        self.assertTrue(os.path.isfile(out_rbd))
        with open(out_rbd, 'rb') as f:
            newrbd: RigidBodyDynamics = pickle.load(f)
        self.assertIsNotNone(newrbd.regressor_linear, "The regressor_linear is not set")
        self.assertIsNotNone(newrbd.params, "The parameters are not set")

        # Compile RD
        self.assertFalse(os.path.isfile(out_rd))
        api.compile_rd(out_rbd, friction_load_model, friction_viscous_powers, out_rd, self.cache)
        self.assertTrue(os.path.isfile(out_rd))

        # Calibrate
        self.assertFalse(os.path.isfile(calibration_out))
        self.assertFalse(os.path.isfile(params_out))
        api.calibrate(out_rd, data_file, gravity, params_out, calibration_out, plotting=False)
        self.assertTrue(os.path.isfile(calibration_out))
        self.assertTrue(os.path.isfile(params_out))

        # Predict
        prediction_path = from_cache(prediction)
        self.assertFalse(os.path.isfile(prediction_path))
        api.predict(calibration_out, data_file, gravity, prediction)
        self.assertTrue(os.path.isfile(prediction_path))


if __name__ == '__main__':
    unittest.main()
