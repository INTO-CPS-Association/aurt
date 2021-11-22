import argparse
from logging.config import fileConfig
import logging
import numpy as np
import os.path
import aurt.api as api
from aurt.caching import PersistentPickleCache, clear_cache_dir
from aurt.messages import CONTENT_OVERWRITTEN


def setup_logger(args):
    if args.logger_config is not None:
        fileConfig(args.logger_config)
    else:
        logging.basicConfig(level=logging.WARNING)
    return logging.getLogger("aurt")


def compile_rbd(args):
    l = setup_logger(args)
    l.info("Compiling rigid body dynamics model.")

    if args.mdh[-4:] != ".csv":
        raise ValueError(f"The provided mdh file {args.mdh} is not a CSV file. AURT only supports CSV files. Please provide a CSV file.")
    if not os.path.isfile(args.mdh):
        raise OSError(f"The mdh file {args.mdh} could not be located. Please specify a valid filename (csv).")
    output_path = args.out
    if os.path.isfile(output_path):
        l.warning(CONTENT_OVERWRITTEN % {"description": "rigid body dynamics", "filepath": output_path})

    mdh_path = args.mdh
    plotting = args.plot
    cache = PersistentPickleCache(args.cache)
    api.compile_rbd(mdh_path, output_path, plotting, cache)


def compile_rd(args):
    l = setup_logger(args)
    l.info("Compiling robot dynamics model.")
    l.debug(f"Viscous friction powers: {args.friction_viscous_powers}.")

    if not os.path.isfile(args.model_rbd):
        raise ValueError(f"The rigid body dynamics file {args.model_rbd} could not be located. Please specify a valid filename.")

    if os.path.isfile(args.out):
        l.warning(CONTENT_OVERWRITTEN % {"description": "robot dynamics", "filepath": args.out})

    if len(set(args.friction_viscous_powers)) != len(list(args.friction_viscous_powers)):
        raise ValueError(f"The viscous friction powers must be a set of unique integers.")

    model_rbd_path = args.model_rbd
    friction_torque_model = args.friction_torque_model
    friction_viscous_powers = args.friction_viscous_powers
    #friction_hysteresis_model = args.friction_hysteresis_model # saved for later implementation
    output_path = args.out

    l.debug(f"Using folder {args.cache} as cache.")
    cache = PersistentPickleCache(args.cache)
    api.compile_rd(model_rbd_path, friction_torque_model, friction_viscous_powers, output_path, cache)


def calibrate(args):
    l = setup_logger(args)
    l.info("Calibrating robot dynamics model.")

    gravity_vector_type = {True if isinstance(g,float) or isinstance(g,int) else False for g in args.gravity}
    if False in gravity_vector_type:
        raise TypeError(f"The given gravity vector is not a float, nor an integer. Please provide a valid gravity vector.")

    if not os.path.isfile(args.model):
        raise ValueError(f"The robot dynamics file {args.model} could not be located. Please specify a valid filename.")

    if not os.path.isfile(args.data):
        raise ValueError(f"The data file {args.data} could not be located. Please specify a valid filename.")

    if os.path.isfile(args.out_params):
        l.warning(CONTENT_OVERWRITTEN % {"description": "parameters", "filepath": args.out_params})

    if os.path.isfile(args.out_calibrated_model):
        l.warning(CONTENT_OVERWRITTEN % {"description": "calibration model", "filepath": args.out_calibrated_model})

    model_path = args.model
    gravity = args.gravity
    data_path = args.data
    params_path = args.out_params
    calibrated_model_path = args.out_calibrated_model
    plotting = args.plot
    
    api.calibrate(model_path, data_path, gravity, params_path, calibrated_model_path, plotting)


def predict(args):
    l = setup_logger(args)
    l.info("Predicting robot current.")

    if not os.path.isfile(args.model):
        raise ValueError(f"The robot calibration file {args.model} could not be located. Please specify a valid filename.")
    
    if not os.path.isfile(args.data):
        raise ValueError(f"The data file {args.data} could not be located. Please specify a valid filename.")
    
    gravity_vector_type = {True if isinstance(g,float) or isinstance(g,int) else False for g in args.gravity}
    if False in gravity_vector_type:
        raise TypeError(f"The given gravity vector is not a float, nor an integer. Please provide a valid gravity vector.")

    if os.path.isfile(args.out):
        l.warning(CONTENT_OVERWRITTEN % {"description": "output prediction", "filepath": args.out})

    model_path = args.model
    data_path = args.data
    gravity = np.array(args.gravity)
    output_path = args.out
    
    api.predict(model_path, data_path, gravity, output_path)


def calibrate_validate(args):
    l = setup_logger(args)
    l.info("Calibrating and validating robot dynamics model.")

    if not os.path.isfile(args.model):
        raise ValueError(f"The robot dynamics file {args.model} could not be located. Please specify a valid filename.")
    
    if not os.path.isfile(args.data):
        raise ValueError(f"The data file {args.data} could not be located. Please specify a valid filename.")
    
    gravity_vector_type = {True if isinstance(g,float) or isinstance(g,int) else False for g in args.gravity}
    if False in gravity_vector_type:
        raise TypeError(f"The given gravity vector is not a float, nor an integer. Please provide a valid gravity vector.")

    if os.path.isfile(args.out_params):
        l.warning(CONTENT_OVERWRITTEN % {"description": "parameters", "filepath": args.out_params})

    if os.path.isfile(args.out_prediction):
        l.warning(CONTENT_OVERWRITTEN % {"description": "output prediction", "filepath": args.out_prediction})

    if os.path.isfile(args.out_calibrated_model):
        l.warning(CONTENT_OVERWRITTEN % {"description": "calibration model", "filepath": args.out_calibrated_model})

    if not (0.1 < args.calibration_data_rel < 0.9):
        raise ValueError(f"The calibration data rel value is not within the limits of 0.1 and 0.9, it is {args.calibration_data_rel}. Please provide a valid value.")

    model_path = args.model
    data_path = args.data
    gravity = np.array(args.gravity)
    calbrated_model_path = args.out_calibrated_model
    calibration_data_rel = args.calibration_data_rel
    plotting = args.plot
    params_path = args.out_params
    output_prediction_path = args.out_prediction
    
    api.calibrate_validate(model_path, data_path, gravity, calibration_data_rel, params_path, calbrated_model_path, output_prediction_path, plotting)


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
                                    help="Modified Denavit-Hartenberg (MDH) parameters file (csv)")

    compile_rbd_parser.add_argument('--out', required=True,
                                    help="Path of outputted rigid-body dynamics model (pickle).")

    compile_rbd_parser.add_argument('--cache', required=False, default="./cache",
                                    help="Path of folder that is used for temporary storage of results.")

    compile_rbd_parser.add_argument('--plot', action="store_true", default=False)

    compile_rbd_parser.set_defaults(command=compile_rbd)
    return compile_rbd_parser

def _create_compile_rd_parser(subparsers):
    compile_rd_parser = subparsers.add_parser("compile-rd")

    compile_rd_parser.add_argument('--model-rbd', required=True,
                                       help="The rigid-body dynamics model created with the compile-rbd command.")

    compile_rd_parser.add_argument('--friction-torque-model', required=True,
                                       choices=["none", "square", "absolute"],
                                       help="The friction/torque model.")

    compile_rd_parser.add_argument('--friction-viscous-powers', required=True, nargs="+",
                                       type=int,
                                       metavar='R',
                                       help="The viscous friction polynomial powers.")

    compile_rd_parser.add_argument('--cache', required=False, default="cache",
                                    help="Path of folder that is used for temporary storage of results.")

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
    
    calibrate_parser.add_argument('--gravity', required=True,
                                  nargs=3,
                                  type=float,
                                  metavar='R',
                                  help="Components (x, y, and z, respectively) of gravity vector, e.g. <0 0 -9.81>.")

    calibrate_parser.add_argument('--out-params',
                                    help="The resulting parameter values (csv).")

    calibrate_parser.add_argument('--out-calibrated-model',
                                    help="Path of the outputted calibrated robot dynamics model (pickle).")

    calibrate_parser.add_argument('--plot', action="store_true", default=False)

    calibrate_parser.set_defaults(command=calibrate)
    return calibrate_parser

def _create_predict_parser(subparsers):
    predict_parser = subparsers.add_parser("predict")

    predict_parser.add_argument('--model', required=True,
                                    help="The calibrated robot dynamics model created with the calibrate command.")

    predict_parser.add_argument('--data', required=True,
                                    help="The measured data (csv).")
    
    predict_parser.add_argument('--gravity', required=True,
                                nargs=3,
                                type=float,
                                metavar='R',
                                help="Components (x, y, and z, respectively) of gravity vector, e.g. <0 0 -9.81>.")

    predict_parser.add_argument('--out', required=True,
                                    help="Path of outputted prediction values (csv).")

    predict_parser.set_defaults(command=predict)
    return predict_parser

def _create_calibrate_validate_parser(subparsers):
    calibrate_validate_parser = subparsers.add_parser("calibrate-validate")

    calibrate_validate_parser.add_argument('--model', required=True,
                                           help="The robot dynamics model created with the compile-rd command.")

    calibrate_validate_parser.add_argument('--data', required=True, help="The measured data (csv).")

    calibrate_validate_parser.add_argument('--gravity', required=True,
                                           nargs=3,
                                           type=float,
                                           metavar='R',
                                           help="Components (x, y, and z, respectively) of gravity vector, e.g. <0 0 -9.81>.")

    calibrate_validate_parser.add_argument('--calibration-data-rel', required=True, type=float,
                                           help="The relative fraction of the dataset to be used for calibration. The value should be in the range [0.1; 0.9].")

    calibrate_validate_parser.add_argument('--out-params', help="The resulting parameter values (csv).")

    calibrate_validate_parser.add_argument('--out-calibrated-model',
                                           help="Path of the outputted calibrated robot dynamics model (pickle).")

    calibrate_validate_parser.add_argument('--out-prediction', required=True,
                                           help="Path of outputted prediction values (csv).")

    calibrate_validate_parser.add_argument('--plot', action="store_true", default=False)

    calibrate_validate_parser.set_defaults(command=calibrate_validate)
    return calibrate_validate_parser


def main():
    create_cmd_parser()


if __name__ == '__main__':
    main()
