import logging
import os.path
import pickle
import unittest

from aurt import api
from aurt.caching import PersistentPickleCache, clear_cache_dir
from aurt.file_system import from_project_root, from_cache
from aurt.rigid_body_dynamics import RigidBodyDynamics


class URExampleTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Runs when class is loaded.
        """
        cls.cache_dir = from_project_root('cache')
        clear_cache_dir(cls.cache_dir)
        cls.cache = PersistentPickleCache(cls.cache_dir)
        logging.basicConfig(level=logging.DEBUG)

    def test_ur5e_dh(self):
        self.assertEqual(0, len(os.listdir(self.cache_dir)))
        l = logging.getLogger("Tests")
        mdh_path = str(from_project_root("resources/robot_parameters/ur5e_dh.csv"))
        gravity = [0.0, 6.937, -6.937]
        out_rbd = from_cache("rbd_ur5e.pickle")
        out_rd = from_cache("rd_ur5e.pickle")
        friction_load_model = "square"
        friction_viscous_powers = [2, 1, 4]
        data_file = str(from_project_root("resources/Dataset/ur5e_45degX_aurt_demo_1/ur5e_45degX_aurt_demo_1.csv"))
        params_out = from_cache("ur5e_parameters.csv")
        calibration_out = from_cache("rc_ur5e.pickle")
        prediction = from_cache("prediction_ur5e.csv")

        # Compile RBD
        l.info("Compiling rigid-body dynamics...")
        self.assertFalse(os.path.isfile(out_rbd))
        api.compile_rbd(mdh_path, out_rbd, False, self.cache)
        self.assertTrue(os.path.isfile(out_rbd))
        with open(out_rbd, 'rb') as f:
            newrbd: RigidBodyDynamics = pickle.load(f)
        self.assertIsNotNone(newrbd.regressor(), "The regressor is not set")
        self.assertIsNotNone(newrbd.parameters, "The parameters are not set")

        # Compile RD
        l.info("Compile robot dynamics...")
        self.assertFalse(os.path.isfile(out_rd))
        api.compile_rd(out_rbd, friction_load_model, friction_viscous_powers, out_rd, self.cache)
        self.assertTrue(os.path.isfile(out_rd))

        # Calibrate
        l.info("Calibrate")
        self.assertFalse(os.path.isfile(calibration_out))
        self.assertFalse(os.path.isfile(params_out))
        api.calibrate(out_rd, data_file, gravity, params_out, calibration_out, plotting=False)
        self.assertTrue(os.path.isfile(calibration_out))
        self.assertTrue(os.path.isfile(params_out))

        # Predict
        l.info("Validating robot dynamics")
        prediction_path = from_cache(prediction)
        self.assertFalse(os.path.isfile(prediction_path))
        api.predict(calibration_out, data_file, gravity, prediction)
        self.assertTrue(os.path.isfile(prediction_path))


if __name__ == '__main__':
    unittest.main()
