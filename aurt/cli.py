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

    mdh_path = ""
    gravity = np.array(args.gravity)
    output_path = ""

    api.compile_rbd(mdh_path, gravity, output_path)

def compile_jointd(args):
    l = setup_logger(args)
    l.info("Compiling robot dynamics model.")

    l.debug(f"Viscous friction powers: {args.friction_viscous_powers}.")
    # TODO: Do some error checking on the user provided parameters, convert their types, check that files exists
    #  (or that they will not be overwritten) etc.

    model_rbd_path = ""
    friction_load_model = ""
    friction_viscous_powers = args.friction_viscous_powers
    friction_hysteresis_model = args.friction_hysteresis_model
    output_path = args.out

    api.compile_jointd(model_rbd_path, friction_load_model, friction_viscous_powers, friction_hysteresis_model, output_path)

def calibrate(args):
    l = setup_logger(args)
    l.info("Calibrating robot dynamics model.")

    # TODO: Do some error checking on the user provided parameters, convert their types, check that files exists
    #  (or that they will not be overwritten) etc.

    model_path = ""
    data_path = ""
    reduced = ""
    output_path = ""
    
    api.calibrate(model_path, data_path, reduced, output_path)



def main():
    # Command parser
    args_parser = argparse.ArgumentParser(add_help=True)

    args_parser.add_argument('--logger-config', type=open,
                             help="Logger configuration file.")
    # args_parser.add_argument('command', choices=['compile-rbd', 'compile-jointd', 'calibrate', 'predict'])
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

    ## compile_jointd
    compile_jointd_parser = subparsers.add_parser("compile-jointd")

    compile_jointd_parser.add_argument('--model-rbd', required=True,
                                       help="The rigid body dynamics model created with the compile-rbd command.")

    compile_jointd_parser.add_argument('--friction-load-model', required=True,
                                       choices=["none", "square", "absolute"],
                                       help="The friction load model.")

    compile_jointd_parser.add_argument('--friction-viscous-powers', required=True, nargs="+",
                                       type=float,
                                       metavar='R',
                                       help="The viscous friction polynomial powers.")

    compile_jointd_parser.add_argument('--friction-hysteresis-model', required=True,
                                       choices=["sign", "maxwells"],
                                       help="The friction hysteresis model.")

    compile_jointd_parser.add_argument('--out', required=True,
                                       help="Path of outputted robot dynamics model (pickle).")

    compile_jointd_parser.set_defaults(command=compile_jointd)

    ## calibrate
    calibrate_parser = subparsers.add_parser("calibrate")

    calibrate_parser.add_argument('--model', required=True,
                                    help="The robot dynamics model created with the compile-jointd command.")

    calibrate_parser.add_argument('--data', required=True,
                                    help="The measured data (csv).")

    calibrate_parser.add_argument('--out-reduced-params',
                                    help="The resulting reduced parameter values (csv).")

    calibrate_parser.add_argument('--out-full-params',
                                    help="The resulting full parameter values (csv).")

    calibrate_parser.set_defaults(command=calibrate)

    # Force help display when error occurrs. See https://stackoverflow.com/questions/3636967/python-argparse-how-can-i-display-help-automatically-on-error
    args_parser.usage = args_parser.format_help().replace("usage: ", "")

    args = args_parser.parse_args()

    mdh_path = "resources/robot_parameters/ur3e_params.csv"
    gravity = [0, 0, -9.81]
    output_path = "rigid-body_dynamics.pickle"
    api.compile_rbd(mdh_path, gravity, output_path)



    #args.command(args)


if __name__ == '__main__':
    main()
