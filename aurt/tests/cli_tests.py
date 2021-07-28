import unittest
import pickle
from pathlib import Path
from shutil import rmtree

from aurt.cli import *
from aurt.cli import _init_cmd_parser, _create_compile_rbd_parser, _create_calibrate_parser, _create_compile_rd_parser, _create_predict_parser
from aurt.file_system import from_cache

#### TODO:
##    o add tests for compile-rd
##    o add tests for calibrate
##    o add tests for predict

class CLITests(unittest.TestCase):
    
    # The naming of the test cases control the order of the tests

    def init_cache_dir(self):
        # delete cache folder before starting tests
        new_dir = Path(__file__).parent.parent.parent.joinpath('cache')
        if Path(new_dir).is_dir():
            rmtree(Path(new_dir))

    def init_rbd(self):
        self.init_cache_dir()
        mdh_filename = str(Path("resources","robot_parameters", "two_link_model.csv"))
        out_filename = "out_rbd"
        gravity = [0.0, -9.81, 0.0]
        parser = self.set_compile_rbd_arguments(
            mdh_filename,
            gravity,
            out_filename
        )
        compile_rbd(parser)

    def init_rd(self):
        self.init_rbd()
        model_rbd = "out_rbd"
        friction_load_model = "square"
        friction_viscous_powers = [2,1,4]
        out = "out_rd"
        parser = self.set_compile_rd_arguments(
            model_rbd,
            friction_load_model,
            friction_viscous_powers,
            out
        )
        compile_rd(parser)
    
    def set_compile_rbd_arguments(self, mdh, gravity, out, logger_config=None):
        subparsers,_ = _init_cmd_parser()
        compile_rbd_parser = _create_compile_rbd_parser(subparsers)
        compile_rbd_parser.mdh = mdh
        compile_rbd_parser.gravity = gravity
        compile_rbd_parser.out = out
        compile_rbd_parser.logger_config = logger_config
        return compile_rbd_parser

    def set_compile_rd_arguments(self,model_rbd,friction_load_model,friction_viscous_powers,out, logger_config=None):
        subparsers,_ = _init_cmd_parser()
        compile_rd_parser = _create_compile_rd_parser(subparsers)
        compile_rd_parser.model_rbd = model_rbd
        compile_rd_parser.friction_load_model = friction_load_model
        compile_rd_parser.friction_viscous_powers = friction_viscous_powers
        compile_rd_parser.out = out
        compile_rd_parser.logger_config = logger_config
        return compile_rd_parser

    def set_calibrate_arguments(self,model, data, out_params, out_calibration_model, logger_config=None):
        subparsers,_ = _init_cmd_parser()
        calibrate_parser = _create_calibrate_parser(subparsers)
        calibrate_parser.model = model
        calibrate_parser.data = data
        calibrate_parser.out_params = out_params
        calibrate_parser.out_calibration_model = out_calibration_model
        calibrate_parser.logger_config = logger_config
        return calibrate_parser

    def test01_compile_rbd_args_cli_correct(self):
        mdh_filename = str(Path("resources","robot_parameters", "two_link_model.csv"))
        out_filename = "out_rbd"
        gravity = [0.0, 0.0, -9.81]
        parser = self.set_compile_rbd_arguments(
            mdh_filename,
            gravity,
            out_filename
        )
        compile_rbd(parser)

        out_filename = from_cache(out_filename + ".pickle")
        with open(out_filename, 'rb') as f:
            out_rbd = pickle.load(f)
        self.assertTrue(out_rbd != None)

    def test02_compile_rbd_args_cli_file_not_csv(self):
        mdh_filename = str(Path("resources","robot_parameters", "two_link_model"))
        gravity = [0.0, 0.0, -9.81]
        out_filename = "out_rbd"
        parser = self.set_compile_rbd_arguments(
            mdh_filename,
            gravity,
            out_filename
        )
        with self.assertRaises(Exception):
            compile_rbd(parser)

    def test03_compile_rbd_args_cli_file_not_found(self):
        mdh_filename = str(Path("resources","robot_parameters", "wrong_name.csv"))
        gravity = [0.0, 0.0, -9.81]
        out_filename = "out_rbd"
        parser = self.set_compile_rbd_arguments(
            mdh_filename,
            gravity,
            out_filename
        )
        with self.assertRaises(OSError):
            compile_rbd(parser)


    def test04_compile_rbd_args_cli_gravity_not_floats(self):
        mdh_filename = str(Path("resources","robot_parameters", "two_link_model.csv"))
        gravity = ["a","b","c"]
        out_filename = "out_rbd"
        parser = self.set_compile_rbd_arguments(
            mdh_filename,
            gravity,
            out_filename
        )
        with self.assertRaises(TypeError):
            compile_rbd(parser)

    def test05_compile_rbd_args_cli_gravity_ints(self):
        mdh_filename = str(Path("resources","robot_parameters", "two_link_model.csv"))
        gravity = [0,1,0]
        out_filename = "out_rbd"
        compile_rbd_parser = self.set_compile_rbd_arguments(
            mdh_filename,
            gravity,
            out_filename
        )
        compile_rbd(compile_rbd_parser)
        out_filename = from_cache(out_filename + ".pickle")
        with open(out_filename, 'rb') as f:
            out_rbd = pickle.load(f)
        self.assertTrue(out_rbd != None)

    def test06_compile_rbd_args_cli_out_invalid(self):
        mdh_filename = str(Path("resources","robot_parameters", "two_link_model.csv"))
        gravity = ["a","b","c"]
        out_filename = "out_rbd"
        parser = self.set_compile_rbd_arguments(
            mdh_filename,
            gravity,
            out_filename
        )
        with self.assertRaises(TypeError):
            compile_rbd(parser)

    def test07_compile_rd_args_cli_correct(self):
        self.init_rbd()
        model_rbd = "out_rbd"
        friction_load_model = "square"
        friction_viscous_powers = [2,1,4]
        out = "out_rd"
        parser = self.set_compile_rd_arguments(
            model_rbd,
            friction_load_model,
            friction_viscous_powers,
            out
        )
        compile_rd(parser)

        out_filename = from_cache(out + ".pickle")
        with open(out_filename, 'rb') as f:
            out_rd = pickle.load(f)
        self.assertTrue(out_rd != None)

    def test08_calibrate_args_cli_correct(self):
        self.init_rd()
        model = "out_rd"
        data = "aurt/tests/resources/twolink_data.csv"
        out_params = "calibrated_params.csv"
        out_calibration_model = "out_calibration"
        parser = self.set_calibrate_arguments(
            model,
            data,
            out_params,
            out_calibration_model
        )
        calibrate(parser)

        out_filename = from_cache(out_calibration_model + ".pickle")
        with open(out_filename, 'rb') as f:
            out_calibration = pickle.load(f)
        self.assertTrue(out_calibration != None)



if __name__ == '__main__':
    unittest.main()
