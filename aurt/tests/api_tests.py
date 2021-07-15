
import unittest
import pickle
from pathlib import Path
from shutil import rmtree

from aurt.file_system import from_cache
from aurt import api
from aurt.rigid_body_dynamics import RigidBodyDynamics

class APITests(unittest.TestCase):

    def setUp(self) -> None:
        # delete cache folder before starting tests
        new_dir = Path(__file__).parent.parent.parent.joinpath('cache')
        if Path(new_dir).is_dir():
            rmtree(Path(new_dir))
        return super().setUp()

    def test_compile_rbd(self):
        mdh_path = "aurt/tests/resources/two_link_model.csv"
        gravity = [0.0, -9.81, 0.0]
        output_path = "rbd_twolink"

        api.compile_rbd(mdh_path, gravity, output_path)
        filename = from_cache(output_path + ".pickle")
        with open(filename, 'rb') as f:
            rbd_twolink_estimate: RigidBodyDynamics = pickle.load(f)
        with open("aurt/tests/resources/rbd_twolink.pickle", 'rb') as f:
            rbd_twolink_true: RigidBodyDynamics = pickle.load(f)

        self.assertEqual(rbd_twolink_estimate.mdh, rbd_twolink_true.mdh)
        self.assertEqual(rbd_twolink_estimate.regressor_linear, rbd_twolink_true.regressor_linear)
        self.assertEqual(rbd_twolink_estimate.n_params, rbd_twolink_true.n_params)
        self.assertEqual(rbd_twolink_estimate.params, rbd_twolink_true.params)

    def test_compile_rbd_save_class(self):
        # test that class is saved properly
        mdh_path = "aurt/tests/resources/two_link_model.csv"
        gravity = [0.0, -9.81, 0.0]
        output_path = "rbd_twolink"

        api.compile_rbd(mdh_path, gravity, output_path)
        filename = from_cache(output_path + ".pickle")
    
        with open(filename, 'rb') as f:
            newrbd: RigidBodyDynamics = pickle.load(f)

        self.assertIsNotNone(newrbd.regressor_linear, "The regressor_linear is not set")
        self.assertIsNotNone(newrbd.params, "The paramters are not set")


    def test_compile_rd(self):
        model_rbd = "rbd_twolink"
        friction_load_model = "square"
        friction_viscous_powers = [2,1,4]
        output_file = "rd_twolink"

        # Create RigidBodyDynamics Model
        mdh_path = "aurt/tests/resources/two_link_model.csv"
        gravity = [0.0, -9.81, 0.0]
        rbd_output_path = "rbd_twolink"
        api.compile_rbd(mdh_path, gravity, rbd_output_path)

        api.compile_rd(model_rbd, friction_load_model, friction_viscous_powers, output_file)

        filename = from_cache(output_file + ".pickle")
        with open(filename, 'rb') as f:
            rd_twolink_estimate = pickle.load(f)    
        with open("aurt/tests/resources/rd_twolink.pickle", 'rb') as f:
            rd_twolink_true = pickle.load(f)

        self.assertEqual(rd_twolink_estimate.n_joints,rd_twolink_true.n_joints)
        self.assertEqual(rd_twolink_estimate.qdd,rd_twolink_true.qdd)
        self.assertEqual(rd_twolink_estimate.tauJ,rd_twolink_true.tauJ)


    def test_calibrate(self):
        model_rd = "rd_twolink"
        data_file = "aurt/tests/resources/twolink_data.csv"
        params_out = "twolink_params.csv"
        calibration_out = "rc_twolink"

        


if __name__ == '__main__':
    unittest.main()
