import unittest
import pickle
from pathlib import Path
from aurt.file_system import from_cache, from_project_root
from aurt import api
from aurt.rigid_body_dynamics import RigidBodyDynamics

#### TODO:
##    o add tests for calibrate
##    o add tests for predict
from aurt.tests.units import init_cache_dir


class APITests(unittest.TestCase):

    # The naming of the test cases control the order of the tests
    def test01_compile_rbd(self):
        init_cache_dir()
        mdh_path = str(from_project_root("aurt/tests/resources/twolink_model.csv"))
        gravity = [0.0, -9.81, 0.0]
        output_path = "rbd_twolink"
        plotting = False

        api.compile_rbd(mdh_path, gravity, output_path, plotting)
        filename = from_cache(output_path + ".pickle")
        with open(filename, 'rb') as f:
            rbd_twolink_estimate: RigidBodyDynamics = pickle.load(f)
        with open(str(from_project_root(Path("aurt/tests/resources", output_path))) + ".pickle", 'rb') as f:
            rbd_twolink_true: RigidBodyDynamics = pickle.load(f)

        self.assertEqual(rbd_twolink_estimate.mdh, rbd_twolink_true.mdh)
        self.assertEqual(rbd_twolink_estimate.regressor_linear, rbd_twolink_true.regressor_linear)
        self.assertEqual(rbd_twolink_estimate.n_params, rbd_twolink_true.n_params)
        self.assertEqual(rbd_twolink_estimate.params, rbd_twolink_true.params)

    def test02_compile_rbd_save_class(self):
        # test that class is saved properly
        output_path = "rbd_twolink"
        filename = from_cache(output_path + ".pickle")

        with open(filename, 'rb') as f:
            newrbd: RigidBodyDynamics = pickle.load(f)

        self.assertIsNotNone(newrbd.regressor_linear, "The regressor_linear is not set")
        self.assertIsNotNone(newrbd.params, "The paramters are not set")

    def test03_compile_rd(self):
        model_rbd = "rbd_twolink"
        friction_load_model = "square"
        friction_viscous_powers = [2, 1, 4]
        output_file = "rd_twolink"

        api.compile_rd(model_rbd, friction_load_model, friction_viscous_powers, output_file)

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
        params_out = "twolink_params.csv"
        calibration_out = "rc_twolink"
        plotting = False

        api.calibrate(model_rd, data_file, params_out, calibration_out, plotting)

    def test05_predict(self):
        model = "rc_twolink"
        data = str(from_project_root("aurt/tests/resources/twolink_data.csv"))
        prediction = "out_predict.csv"

        api.predict(model, data, prediction)

    def test06_plot_kinematics(self):
        mdh_path = str(from_project_root("aurt/tests/resources/twolink_model.csv"))
        gravity = [0.0, -9.81, 0.0]
        output_path = "rbd_twolink"
        plotting = True

        api.compile_rbd(mdh_path, gravity, output_path, plotting, block=False)


if __name__ == '__main__':
    unittest.main()
