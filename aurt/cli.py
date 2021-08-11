import argparse
from logging.config import fileConfig
import logging
import numpy as np
import os.path
import aurt.api as api
from aurt.file_system import from_cache

def setup_logger(args):
    if args.logger_config is not None:
        fileConfig(args.logger_config)
    else:
        logging.basicConfig(level=logging.INFO)
    return logging.getLogger("aurt")


def compile_rbd(args):
    l = setup_logger(args)
    l.info("Compiling rigid body dynamics model.")
    l.debug(f"Gravity vector: {args.gravity}")

    if args.mdh[-4:] != ".csv":
        raise Exception(f"The provided mdh file {args.mdh} is not a CSV file. AURT only supports CSV files. Please provide a CSV file.")
    if not os.path.isfile(args.mdh):
        raise OSError(f"The mdh file {args.mdh} could not be located. Please specify a valid filename (csv).")
    gravity_vector_type = {True if isinstance(g,float) or isinstance(g,int) else False for g in args.gravity}
    if False in gravity_vector_type:
        raise TypeError(f"The given gravity vector is not a float, nor an integer. Please provide a valid gravity vector.")
    filename = from_cache(args.out + ".pickle")
    if os.path.isfile(filename):
        l.warning(f"The rigid body dynamics file {filename} already exists, and its content will be overwritten.")
    
    mdh_path = args.mdh
    gravity = np.array(args.gravity)
    output_path = args.out
    plotting = args.plot

    api.compile_rbd(mdh_path, gravity, output_path, plotting)


def compile_rd(args):
    l = setup_logger(args)
    l.info("Compiling robot dynamics model.")
    l.debug(f"Viscous friction powers: {args.friction_viscous_powers}.")

    filename = from_cache(args.model_rbd + ".pickle")
    if not os.path.isfile(filename):
        raise Exception(f"The rigid body dynamics file {filename} could not be located. Please specify a valid filename.")

    filename = from_cache(args.out + ".pickle")
    if os.path.isfile(filename):
        l.warning(f"The robot dynamics filename {args.out} already exists, and its content will be overwritten.")

    model_rbd_path = args.model_rbd
    friction_load_model = args.friction_load_model
    friction_viscous_powers = args.friction_viscous_powers
    #friction_hysteresis_model = args.friction_hysteresis_model # saved for later implementation
    output_path = args.out

    api.compile_rd(model_rbd_path, friction_load_model, friction_viscous_powers, output_path)


def calibrate(args):
    l = setup_logger(args)
    l.info("Calibrating robot dynamics model.")

    filename = from_cache(args.model + ".pickle")
    if not os.path.isfile(filename):
        raise Exception(f"The robot dynamics file {filename} could not be located. Please specify a valid filename.")

    if not os.path.isfile(args.data):
        raise Exception(f"The data file {args.data} could not be located. Please specify a valid filename.")

    if os.path.isfile(args.out_params):
        l.warning(f"The parameters filename {args.out_params} already exists, and its content will be overwritten.")

    filename = from_cache(args.out_calibration_model + ".pickle")
    if os.path.isfile(filename):
        l.warning(f"The calibration model filename {args.out_calibration_model} already exists, and its content will be overwritten.")

    model_path = args.model
    data_path = args.data
    params_path = args.out_params
    calbration_model_path = args.out_calibration_model
    plotting = args.plot
    
    api.calibrate(model_path, data_path, params_path, calbration_model_path, plotting)


def predict(args):
    l = setup_logger(args)
    l.info("Predicting robot current.")

    filename = from_cache(args.model + ".pickle")
    if not os.path.isfile(filename):
        raise Exception(f"The robot calibration file {filename} could not be located. Please specify a valid filename.")
    
    if not os.path.isfile(args.data):
        raise Exception(f"The data file {args.data} could not be located. Please specify a valid filename.")

    filename = from_cache(args.prediction)
    if os.path.isfile(filename):
        l.warning(f"The output prediction file {filename} already exists, and its content will be overwritten.")

    model_path = args.model
    data_path = args.data
    output_path = args.prediction
    
    api.predict(model_path, data_path, output_path)


def calibrate_validate(args):
    l = setup_logger(args)
    l.info("Calibrating and validating robot dynamics model.")

    filename = from_cache(args.model + ".pickle")
    if not os.path.isfile(filename):
        raise Exception(f"The robot dynamics file {filename} could not be located. Please specify a valid filename.")
    
    if not os.path.isfile(args.data):
        raise Exception(f"The data file {args.data} could not be located. Please specify a valid filename.")

    if os.path.isfile(args.out_params):
        l.warning(f"The parameters filename {args.out_params} already exists, and its content will be overwritten.")

    filename = from_cache(args.output_prediction)
    if os.path.isfile(filename):
        l.warning(f"The output prediction file {filename} already exists, and its content will be overwritten.")

    filename = from_cache(args.out_calibration_model + ".pickle")
    if os.path.isfile(filename):
        l.warning(f"The calibration model filename {args.out_calibration_model} already exists, and its content will be overwritten.")


    if not (0.1 < args.calibration_data_rel < 0.9):
        raise Exception(f"The calibration data rel value is not within the limits of 0.1 and 0.9, it is {args.calibration_data_rel}. Please provide a valid value.")

    model_path = args.model
    data_path = args.data
    calbration_model_path = args.out_calibration_model
    calibration_data_rel = args.calibration_data_rel # TODO: make sure to check that it is between 0 and 1
    plotting = args.plot
    params_path = args.out_params
    output_prediction_path = args.output_prediction
    
    api.calibrate_validate(model_path, data_path, calibration_data_rel, params_path, calbration_model_path, output_prediction_path, plotting)


def create_cmd_parser():
    # Command parser
    subparsers, args_parser = _init_cmd_parser()

    # sub commands
    ## compile_rbd
    _create_compile_rbd_parser(subparsers)
    ## compile_rd
    _create_compile_rd_parser(subparsers)
    ## calibrate
    _create_calibrate_parser(subparsers)
    ## predict
    _create_predict_parser(subparsers)
    ## validate
    _create_calibrate_validate_parser(subparsers)

    # Force help display when error occurrs. See https://stackoverflow.com/questions/3636967/python-argparse-how-can-i-display-help-automatically-on-error
    args_parser.usage = args_parser.format_help().replace("usage: ", "")

    args = args_parser.parse_args()

    args.command(args)

def _init_cmd_parser():
    args_parser = argparse.ArgumentParser(add_help=True)

    args_parser.add_argument('--logger-config', type=open,
                             help="Logger configuration file.")
    # args_parser.add_argument('command', choices=['compile-rbd', 'compile-rd', 'calibrate', 'predict'])
    # args_parser.add_argument('-h', help="Show help.", action="store_true")

    subparsers = args_parser.add_subparsers(
        help="Command to execute. Type 'aurt CMD --help' for more help about command CMD.")
    return subparsers, args_parser

def _create_compile_rbd_parser(subparsers):
    compile_rbd_parser = subparsers.add_parser("compile-rbd")

    compile_rbd_parser.add_argument('--mdh', required=True,
                                    help="Modified Denavit Hartenberg (MDH) parameters file (csv)")

    compile_rbd_parser.add_argument('--gravity', required=True,
                                    nargs=3,
                                    type=float,
                                    metavar='R',
                                    help="Gravity vector. Ex: 0 0 -9.81")

    compile_rbd_parser.add_argument('--out', required=True,
                                    help="Path of outputted rigid body dynamics model (pickle).")

    compile_rbd_parser.add_argument('--plot', action="store_true", default=False)

    compile_rbd_parser.set_defaults(command=compile_rbd)
    return compile_rbd_parser

def _create_compile_rd_parser(subparsers):
    compile_rd_parser = subparsers.add_parser("compile-rd")

    compile_rd_parser.add_argument('--model-rbd', required=True,
                                       help="The rigid body dynamics model created with the compile-rbd command.")

    compile_rd_parser.add_argument('--friction-load-model', required=True,
                                       choices=["none", "square", "absolute"],
                                       help="The friction load model.")

    compile_rd_parser.add_argument('--friction-viscous-powers', required=True, nargs="+",
                                       type=float,
                                       metavar='R',
                                       help="The viscous friction polynomial powers.")

    # compile_rd_parser.add_argument('--friction-hysteresis-model', required=True,
    #                                    choices=["sign", "maxwells"],
    #                                    help="The friction hysteresis model.")

    compile_rd_parser.add_argument('--out', required=True,
                                       help="Path of outputted robot dynamics model (pickle).")

    compile_rd_parser.set_defaults(command=compile_rd)
    return compile_rd_parser

def _create_calibrate_parser(subparsers):
    calibrate_parser = subparsers.add_parser("calibrate")

    calibrate_parser.add_argument('--model', required=True,
                                    help="The robot dynamics model created with the compile-rd command.")

    calibrate_parser.add_argument('--data', required=True,
                                    help="The measured data (csv).")

    calibrate_parser.add_argument('--out-params',
                                    help="The resulting parameter values (csv).")

    calibrate_parser.add_argument('--out-calibration-model',
                                    help="Path of the outputted robot calibration model (pickle).")

    calibrate_parser.add_argument('--plot', action="store_true", default=False)

    calibrate_parser.set_defaults(command=calibrate)
    return calibrate_parser

def _create_predict_parser(subparsers):
    predict_parser = subparsers.add_parser("predict")

    predict_parser.add_argument('--model', required=True,
                                    help="The calibration model created with the calibrate command.")

    predict_parser.add_argument('--data', required=True,
                                    help="The measured data (csv).")

    predict_parser.add_argument('--prediction', required=True,
                                    help="Path of outputted prediction values (csv).")

    predict_parser.set_defaults(command=predict)
    return predict_parser

def _create_calibrate_validate_parser(subparsers):
    calibrate_validate_parser = subparsers.add_parser("calibrate-validate")

    calibrate_validate_parser.add_argument('--model', required=True,
                                           help="The robot dynamics model created with the compile-rd command.")

    calibrate_validate_parser.add_argument('--data', required=True, help="The measured data (csv).")

    calibrate_validate_parser.add_argument('--calibration-data-rel', required=True, type=float,
                                           help="The relative fraction of the dataset to be used for calibration. The value should be in the range [0.1; 0.9].")

    calibrate_validate_parser.add_argument('--out-params', help="The resulting parameter values (csv).")

    calibrate_validate_parser.add_argument('--out-calibration-model',
                                           help="Path of the outputted robot calibration model (pickle).")

    calibrate_validate_parser.add_argument('--output-prediction', required=True,
                                           help="Path of outputted prediction values (csv).")

    calibrate_validate_parser.add_argument('--plot', action="store_true", default=False)

    calibrate_validate_parser.set_defaults(command=calibrate_validate)
    return calibrate_validate_parser

def main():
   create_cmd_parser()


if __name__ == '__main__':
    main()
