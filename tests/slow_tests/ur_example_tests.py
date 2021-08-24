import os.path
import pickle
import unittest

from aurt import api
from aurt.file_system import from_project_root, from_cache
from aurt.rigid_body_dynamics import RigidBodyDynamics
from aurt.tests.units import init_cache_dir


class URExampleTests(unittest.TestCase):

    def test_ur5e_dh(self):
        init_cache_dir()
        mdh_path = str(from_project_root("resources/robot_parameters/ur5e_dh.csv"))
        gravity = [0.0, 6.937, -6.937]
        out_rbd = "rigid_body_dynamics"
        out_rd = "rd_twolink"
        friction_load_model = "square"
        friction_viscous_powers = [2, 1, 4]
        data_file = str(from_project_root("resources/Dataset/ur5e_45degX_aurt_demo_1/ur5e_45degX_aurt_demo_1.csv"))
        params_out = "ur5e_params.csv"
        calibration_out = "rc_ur5e"
        prediction = "out_predict.csv"

        # Compile RBD
        pickle_rbd = from_cache(out_rbd + ".pickle")
        self.assertFalse(os.path.isfile(pickle_rbd))
        api.compile_rbd(mdh_path, gravity, out_rbd, plotting=False)
        self.assertTrue(os.path.isfile(pickle_rbd))
        with open(pickle_rbd, 'rb') as f:
            newrbd: RigidBodyDynamics = pickle.load(f)
        self.assertIsNotNone(newrbd.regressor_linear, "The regressor_linear is not set")
        self.assertIsNotNone(newrbd.params, "The parameters are not set")

        # Compile RD
        pickle_rd = from_cache(out_rd + ".pickle")
        self.assertFalse(os.path.isfile(pickle_rd))
        api.compile_rd(out_rbd, friction_load_model, friction_viscous_powers, out_rd)
        self.assertTrue(os.path.isfile(pickle_rd))

        # Calibrate
        pickle_calibration = from_cache(calibration_out + ".pickle")
        params_path = from_cache(params_out)
        self.assertFalse(os.path.isfile(pickle_calibration))
        self.assertFalse(os.path.isfile(params_path))
        api.calibrate(out_rd, data_file, params_out, calibration_out, plotting=False)
        self.assertTrue(os.path.isfile(pickle_calibration))
        self.assertTrue(os.path.isfile(params_path))

        # Predict
        prediction_path = from_cache(prediction)
        self.assertFalse(os.path.isfile(prediction_path))
        api.predict(calibration_out, data_file, prediction)
        self.assertTrue(os.path.isfile(prediction_path))


if __name__ == '__main__':
    unittest.main()
