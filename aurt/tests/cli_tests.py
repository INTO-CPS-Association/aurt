import unittest
import pickle
from aurt.cli import *
from aurt.cli import _init_cmd_parser, _create_compile_rbd_parser, _create_calibrate_parser, _create_compile_rd_parser, \
    _create_predict_parser
from aurt.file_system import from_cache, load_csv, from_project_root
from aurt.caching import clear_cache_dir


class CLITests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Runs when class is loaded.
        """
        cls.cache_dir = from_project_root('cache')
        clear_cache_dir(cls.cache_dir)
        logging.basicConfig(level=logging.WARNING)

    def init_rbd(self):
        mdh_filename = str(from_project_root("aurt/tests/resources/twolink_dh.csv"))
        out_filename = "out_rbd.pickle"
        cache = self.cache_dir
        plotting = False
        parser = self.set_compile_rbd_arguments(
            mdh_filename,
            out_filename,
            cache,
            plotting
        )
        compile_rbd(parser)

    def init_rd(self):
        self.init_rbd()
        model_rbd = "out_rbd.pickle"
        friction_torque_model = "square"
        friction_viscous_powers = [2, 1, 4]
        out = "out_rd.pickle"
        cache = self.cache_dir
        parser = self.set_compile_rd_arguments(
            model_rbd,
            friction_torque_model,
            friction_viscous_powers,
            out,
            cache
        )
        compile_rd(parser)

    def init_calibrate(self):
        self.init_rd()
        model = "out_rd.pickle"
        data = str(from_project_root("aurt/tests/resources/twolink_data.csv"))
        gravity = [0, -9.81, 0]
        out_params = "twolink_model_params.csv"
        out_calibrated_model = "out_calibrate.pickle"
        cache = self.cache_dir
        plot = False
        parser = self.set_calibrate_arguments(
            model,
            data,
            gravity,
            out_params,
            out_calibrated_model,
            cache,
            plot
        )
        calibrate(parser)

    def set_compile_rbd_arguments(self, mdh, out, cache, plotting, logger_config=None):
        subparsers, _ = _init_cmd_parser()
        compile_rbd_parser = _create_compile_rbd_parser(subparsers)
        compile_rbd_parser.mdh = mdh
        compile_rbd_parser.out = out
        compile_rbd_parser.plot = plotting
        compile_rbd_parser.logger_config = logger_config
        compile_rbd_parser.cache = cache
        return compile_rbd_parser

    def set_compile_rd_arguments(self, model_rbd, friction_torque_model, friction_viscous_powers, out, cache,
                                 logger_config=None):
        subparsers, _ = _init_cmd_parser()
        compile_rd_parser = _create_compile_rd_parser(subparsers)
        compile_rd_parser.model_rbd = model_rbd
        compile_rd_parser.friction_torque_model = friction_torque_model
        compile_rd_parser.friction_viscous_powers = friction_viscous_powers
        compile_rd_parser.out = out
        compile_rd_parser.logger_config = logger_config
        compile_rd_parser.cache = cache
        return compile_rd_parser

    def set_calibrate_arguments(self, model, data, gravity, out_params, out_calibrated_model, cache, plot, logger_config=None):
        subparsers, _ = _init_cmd_parser()
        calibrate_parser = _create_calibrate_parser(subparsers)
        calibrate_parser.model = model
        calibrate_parser.data = data
        calibrate_parser.gravity = gravity
        calibrate_parser.out_params = out_params
        calibrate_parser.out_calibrated_model = out_calibrated_model
        calibrate_parser.cache = cache
        calibrate_parser.plot = plot
        calibrate_parser.logger_config = logger_config
        return calibrate_parser

    def set_predict_arguments(self, model, data, gravity, prediction, cache, logger_config=None):
        subparsers, _ = _init_cmd_parser()
        predict_parser = _create_predict_parser(subparsers)
        predict_parser.model = model
        predict_parser.data = data
        predict_parser.gravity = gravity
        predict_parser.out = prediction
        predict_parser.cache = cache
        predict_parser.logger_config = logger_config
        return predict_parser

    def test01_compile_rbd_args_cli_correct(self):
        mdh_filename = str(from_project_root("aurt/tests/resources/twolink_dh.csv"))
        out_filename = "out_rbd.pickle"
        cache = self.cache_dir
        plotting = False
        parser = self.set_compile_rbd_arguments(
            mdh_filename,
            out_filename,
            cache,
            plotting
        )
        compile_rbd(parser)

        with open(out_filename, 'rb') as f:
            out_rbd = pickle.load(f)

        self.assertTrue(out_rbd is not None)

    def test02_compile_rbd_args_cli_file_not_csv(self):
        mdh_filename = str(from_project_root("aurt/tests/resources/twolink_dh"))
        out_filename = "out_rbd.pickle"
        cache = self.cache_dir
        plotting = False
        parser = self.set_compile_rbd_arguments(
            mdh_filename,
            out_filename,
            cache,
            plotting
        )
        with self.assertRaises(Exception):
            compile_rbd(parser)

    def test03_compile_rbd_args_cli_file_not_found(self):
        mdh_filename = str(from_project_root("aurt/tests/resources/wrong_name.csv"))
        out_filename = "out_rbd.pickle"
        cache = self.cache_dir
        plotting = False
        parser = self.set_compile_rbd_arguments(
            mdh_filename,
            out_filename,
            cache,
            plotting
        )
        with self.assertRaises(OSError):
            compile_rbd(parser)

    def test04_compile_rd_args_cli_correct(self):
        self.init_rbd()
        model_rbd = "out_rbd.pickle"
        friction_torque_model = "square"
        friction_viscous_powers = [2, 1, 4]
        out = "out_rd.pickle"
        cache = self.cache_dir
        parser = self.set_compile_rd_arguments(
            model_rbd,
            friction_torque_model,
            friction_viscous_powers,
            out,
            cache
        )
        compile_rd(parser)

        with open(out, 'rb') as f:
            out_rd = pickle.load(f)
        self.assertTrue(out_rd is not None)

    def test05_calibrate_args_cli_correct(self):
        self.init_rd()
        model = "out_rd.pickle"
        data = str(from_project_root("aurt/tests/resources/twolink_data.csv"))
        gravity = [0, -9.81, 0]
        out_params = "calibrated_params.csv"
        out_calibrated_model = "out_calibration.pickle"
        cache = self.cache_dir
        plot = False
        parser = self.set_calibrate_arguments(
            model,
            data,
            gravity,
            out_params,
            out_calibrated_model,
            cache,
            plot
        )
        calibrate(parser)

        with open(out_calibrated_model, 'rb') as f:
            out_calibration = pickle.load(f)
        self.assertTrue(out_calibration is not None)
    
    def test06_calibrate_args_cli_gravity_not_floats(self):
        self.init_rd()
        model = "out_rd.pickle"
        data = str(from_project_root("aurt/tests/resources/twolink_data.csv"))
        gravity = ["a", "b", "c"]
        out_params = "calibrated_params.csv"
        out_calibrated_model = "out_calibration.pickle"
        cache = self.cache_dir
        plot = False
        parser = self.set_calibrate_arguments(
            model,
            data,
            gravity,
            out_params,
            out_calibrated_model,
            cache,
            plot
        )
        with self.assertRaises(TypeError):
            calibrate(parser)

    def test07_calibrate_args_cli_gravity_ints(self):
        self.init_rd()
        model = "out_rd.pickle"
        data = str(from_project_root("aurt/tests/resources/twolink_data.csv"))
        gravity = [0, -1, 0]
        out_params = "calibrated_params.csv"
        out_calibrated_model = "out_calibration.pickle"
        cache = self.cache_dir
        plot = False
        parser = self.set_calibrate_arguments(
            model,
            data,
            gravity,
            out_params,
            out_calibrated_model,
            cache,
            plot
        )
        calibrate(parser)

        with open(out_calibrated_model, 'rb') as f:
            out_calibration = pickle.load(f)
        self.assertTrue(out_calibration is not None)

    def test08_predict_args_cli_correct(self):
        self.init_calibrate()
        model = "out_calibrate.pickle"
        data = str(from_project_root("aurt/tests/resources/twolink_data.csv"))
        gravity = [0, -9.81, 0]
        out = "out_predict.csv"
        cache = self.cache_dir
        parser = self.set_predict_arguments(
            model,
            data,
            gravity,
            out,
            cache
        )
        predict(parser)
        out_prediction = load_csv(out)
        self.assertFalse(out_prediction.empty)


if __name__ == '__main__':
    unittest.main()
