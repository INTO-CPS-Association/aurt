import unittest
import pickle
from aurt.cli import *
from aurt.cli import _init_cmd_parser, _create_compile_rbd_parser, _create_calibrate_parser, _create_compile_rd_parser, \
    _create_predict_parser
from aurt.file_system import from_cache, load_csv, from_project_root


#### TODO:
##    o add tests for compile-rd
##    o add tests for calibrate
##    o add tests for predict


class CLITests(unittest.TestCase):

    def init_rbd(self):
        mdh_filename = str(from_project_root("aurt/tests/resources/twolink_dh.csv"))
        out_filename = from_cache("out_rbd.pickle")
        plotting = False
        parser = self.set_compile_rbd_arguments(
            mdh_filename,
            out_filename,
            plotting
        )
        compile_rbd(parser)

    def init_rd(self):
        self.init_rbd()
        model_rbd = from_cache("out_rbd.pickle")
        friction_torque_model = "square"
        friction_viscous_powers = [2, 1, 4]
        out = from_cache("out_rd")
        parser = self.set_compile_rd_arguments(
            model_rbd,
            friction_torque_model,
            friction_viscous_powers,
            out
        )
        compile_rd(parser)

    def init_calibrate(self):
        self.init_rd()
        model = from_cache("out_rbd.pickle")
        data = str(from_project_root("aurt/tests/resources/twolink_data.csv"))
        gravity = [0, -9.81, 0]
        out_params = from_cache("twolink_model_params.csv")
        out_calibration_model = from_cache("out_calibrate.pickle")
        plot = False
        parser = self.set_calibrate_arguments(
            model,
            data,
            gravity,
            out_params,
            out_calibration_model,
            plot
        )
        calibrate(parser)

    def set_compile_rbd_arguments(self, mdh, out, plotting, logger_config=None):
        subparsers, _ = _init_cmd_parser()
        compile_rbd_parser = _create_compile_rbd_parser(subparsers)
        compile_rbd_parser.mdh = mdh
        compile_rbd_parser.out = out
        compile_rbd_parser.plot = plotting
        compile_rbd_parser.logger_config = logger_config
        return compile_rbd_parser

    def set_compile_rd_arguments(self, model_rbd, friction_torque_model, friction_viscous_powers, out,
                                 logger_config=None):
        subparsers, _ = _init_cmd_parser()
        compile_rd_parser = _create_compile_rd_parser(subparsers)
        compile_rd_parser.model_rbd = model_rbd
        compile_rd_parser.friction_torque_model = friction_torque_model
        compile_rd_parser.friction_viscous_powers = friction_viscous_powers
        compile_rd_parser.out = out
        compile_rd_parser.logger_config = logger_config
        return compile_rd_parser

    def set_calibrate_arguments(self, model, data, gravity, out_params, out_calibration_model, plot, logger_config=None):
        subparsers, _ = _init_cmd_parser()
        calibrate_parser = _create_calibrate_parser(subparsers)
        calibrate_parser.model = model
        calibrate_parser.data = data
        calibrate_parser.gravity = gravity
        calibrate_parser.out_params = out_params
        calibrate_parser.out_calibration_model = out_calibration_model
        calibrate_parser.plot = plot
        calibrate_parser.logger_config = logger_config
        return calibrate_parser

    def set_predict_arguments(self, model, data, gravity, prediction, logger_config=None):
        subparsers, _ = _init_cmd_parser()
        predict_parser = _create_predict_parser(subparsers)
        predict_parser.model = model
        predict_parser.data = data
        predict_parser.gravity = gravity
        predict_parser.out = prediction
        predict_parser.logger_config = logger_config
        return predict_parser

    def test01_compile_rbd_args_cli_correct(self):
        mdh_filename = str(from_project_root("aurt/tests/resources/twolink_dh.csv"))
        out_filename = from_cache("out_rbd.pickle")
        plotting = False
        parser = self.set_compile_rbd_arguments(
            mdh_filename,
            out_filename,
            plotting
        )
        compile_rbd(parser)

        with open(out_filename, 'rb') as f:
            out_rbd = pickle.load(f)
        self.assertTrue(out_rbd != None)

    def test02_compile_rbd_args_cli_file_not_csv(self):
        mdh_filename = str(from_project_root("aurt/tests/resources/twolink_dh"))
        out_filename = from_cache("out_rbd.pickle")
        plotting = False
        parser = self.set_compile_rbd_arguments(
            mdh_filename,
            out_filename,
            plotting
        )
        with self.assertRaises(Exception):
            compile_rbd(parser)

    def test03_compile_rbd_args_cli_file_not_found(self):
        mdh_filename = str(from_project_root("aurt/tests/resources/wrong_name.csv"))
        out_filename = from_cache("out_rbd.pickle")
        plotting = False
        parser = self.set_compile_rbd_arguments(
            mdh_filename,
            out_filename,
            plotting
        )
        with self.assertRaises(OSError):
            compile_rbd(parser)

    def test04_compile_rd_args_cli_correct(self):
        self.init_rbd()
        model_rbd = from_cache("out_rbd.pickle")
        friction_torque_model = "square"
        friction_viscous_powers = [2, 1, 4]
        out = from_cache("out_rd.pickle")
        parser = self.set_compile_rd_arguments(
            model_rbd,
            friction_torque_model,
            friction_viscous_powers,
            out
        )
        compile_rd(parser)

        with open(out, 'rb') as f:
            out_rd = pickle.load(f)
        self.assertTrue(out_rd is not None)

    def test05_calibrate_args_cli_correct(self):
        self.init_rd()
        model = from_cache("out_rd.pickle")
        data = str(from_project_root("aurt/tests/resources/twolink_data.csv"))
        gravity = [0.7, -0.7, 0.7]
        out_params = from_cache("calibrated_params.csv")
        out_calibration_model = from_cache("out_calibration.pickle")
        plot = False
        parser = self.set_calibrate_arguments(
            model,
            data,
            gravity,
            out_params,
            out_calibration_model,
            plot
        )
        calibrate(parser)

        with open(out_calibration_model, 'rb') as f:
            out_calibration = pickle.load(f)
        self.assertTrue(out_calibration is not None)
    
    def test06_calibrate_args_cli_gravity_not_floats(self):
        self.init_rd()
        model = from_cache("out_rd.pickle")
        data = str(from_project_root("aurt/tests/resources/twolink_data.csv"))
        gravity = ["a", "b", "c"]
        out_params = from_cache("calibrated_params.csv")
        out_calibration_model = from_cache("out_calibration.pickle")
        plot = False
        parser = self.set_calibrate_arguments(
            model,
            data,
            gravity,
            out_params,
            out_calibration_model,
            plot
        )
        with self.assertRaises(TypeError):
            calibrate(parser)

    def test07_calibrate_args_cli_gravity_ints(self):
        self.init_rd()
        model = from_cache("out_rd.pickle")
        data = str(from_project_root("aurt/tests/resources/twolink_data.csv"))
        gravity = [0, 1, 0]
        out_params = from_cache("calibrated_params.csv")
        out_calibration_model = from_cache("out_calibration.pickle")
        plot = False
        parser = self.set_calibrate_arguments(
            model,
            data,
            gravity,
            out_params,
            out_calibration_model,
            plot
        )
        calibrate(parser)

        with open(out_calibration_model, 'rb') as f:
            out_calibration = pickle.load(f)
        self.assertTrue(out_calibration is not None)

    def test08_predict_args_cli_correct(self):
        self.init_calibrate()
        model = from_cache("out_calibrate.pickle")
        data = str(from_project_root("aurt/tests/resources/twolink_data.csv"))  # TODO: change to real prediction data
        gravity = [0, -9.81, 0]
        out = from_cache("out_predict.csv")
        parser = self.set_predict_arguments(
            model,
            data,
            gravity,
            out
        )
        predict(parser)
        out_prediction = load_csv(out)
        self.assertFalse(out_prediction.empty)


if __name__ == '__main__':
    unittest.main()
