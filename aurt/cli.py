import argparse
from logging.config import fileConfig
import logging
import numpy as np
import aurt.api as api


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
    # TODO: Do some error checking on the user provided parameters, convert their types, check that files exists
    #  (or that they will not be overwritten) etc.

    mdh_path = args.mdh
    gravity = np.array(args.gravity)
    output_path = args.out

    api.compile_rbd(mdh_path, gravity, output_path)

def compile_rd(args):
    l = setup_logger(args)
    l.info("Compiling robot dynamics model.")

    l.debug(f"Viscous friction powers: {args.friction_viscous_powers}.")
    # TODO: Do some error checking on the user provided parameters, convert their types, check that files exists
    #  (or that they will not be overwritten) etc.

    model_rbd_path = args.model_rbd
    friction_load_model = args.friction_load_model
    friction_viscous_powers = args.friction_viscous_powers
    friction_hysteresis_model = args.friction_hysteresis_model
    output_path = args.out

    api.compile_rd(model_rbd_path, friction_load_model, friction_viscous_powers, friction_hysteresis_model, output_path)

def calibrate(args):
    l = setup_logger(args)
    l.info("Calibrating robot dynamics model.")

    # TODO: Do some error checking on the user provided parameters, convert their types, check that files exists
    #  (or that they will not be overwritten) etc.

    model_path = args.model
    data_path = args.data
    params_path = args.out_params
    calbration_model_path = args.out_calibration_model
    
    api.calibrate(model_path, data_path, params_path, calbration_model_path)


def predict(args):
    l = setup_logger(args)
    l.info("Predicting robot current.")

    # TODO: Do some error checking on the user provided parameters, convert their types, check that files exists
    #  (or that they will not be overwritten) etc.

    model_path = args.model
    data_path = args.data
    output_path = args.prediction
    
    api.predict(model_path, data_path, output_path)



def main():
    # Command parser
    args_parser = argparse.ArgumentParser(add_help=True)

    args_parser.add_argument('--logger-config', type=open,
                             help="Logger configuration file.")
    # args_parser.add_argument('command', choices=['compile-rbd', 'compile-rd', 'calibrate', 'predict'])
    # args_parser.add_argument('-h', help="Show help.", action="store_true")

    subparsers = args_parser.add_subparsers(
        help="Command to execute. Type 'aurt CMD --help' for more help about command CMD.")

    # sub commands

    ## compile_rbd
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

    compile_rbd_parser.set_defaults(command=compile_rbd)

    ## compile_rd
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

    compile_rd_parser.add_argument('--friction-hysteresis-model', required=True,
                                       choices=["sign", "maxwells"],
                                       help="The friction hysteresis model.")

    compile_rd_parser.add_argument('--out', required=True,
                                       help="Path of outputted robot dynamics model (pickle).")

    compile_rd_parser.set_defaults(command=compile_rd)

    ## calibrate
    calibrate_parser = subparsers.add_parser("calibrate")

    calibrate_parser.add_argument('--model', required=True,
                                    help="The robot dynamics model created with the compile-rd command.")

    calibrate_parser.add_argument('--data', required=True,
                                    help="The measured data (csv).")

    calibrate_parser.add_argument('--out-params',
                                    help="The resulting parameter values (csv).")

    calibrate_parser.add_argument('--out-calibration-model',
                                    help="Path of the outputted robot calibration model (pickle).")

    calibrate_parser.set_defaults(command=calibrate)

    
    ## predict
    predict_parser = subparsers.add_parser("predict")

    predict_parser.add_argument('--model', required=True,
                                    help="The calibration model created with the calibrate command.")

    predict_parser.add_argument('--data', required=True,
                                    help="The measured data (csv).")

    predict_parser.add_argument('--prediction', required=True,
                                    help="Path of outputted prediction values (csv).")

    predict_parser.set_defaults(command=predict)

    # Force help display when error occurrs. See https://stackoverflow.com/questions/3636967/python-argparse-how-can-i-display-help-automatically-on-error
    args_parser.usage = args_parser.format_help().replace("usage: ", "")

    args = args_parser.parse_args()

    args.command(args)


if __name__ == '__main__':
    main()
