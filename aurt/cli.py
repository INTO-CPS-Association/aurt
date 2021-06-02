import argparse
from logging.config import fileConfig
import logging


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


def compile_jointd(args):
    l = setup_logger(args)
    l.info("Compiling joint dynamics model.")
    
    l.debug(f"Viscous friction powers: {args.friction_viscous_powers}.")


def main():
    # Command parser
    args_parser = argparse.ArgumentParser(add_help=True)

    args_parser.add_argument('--logger-config', type=open,
                               help="Logger configuration file.")
    # args_parser.add_argument('command', choices=['compile-rbd', 'compile-jointd', 'calibrate', 'predict'])
    # args_parser.add_argument('-h', help="Show help.", action="store_true")

    subparsers = args_parser.add_subparsers(help="Command to execute. Type 'aurt CMD --help' for more help about command CMD.")

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

    # Force help display when error occurrs. See https://stackoverflow.com/questions/3636967/python-argparse-how-can-i-display-help-automatically-on-error
    args_parser.usage = args_parser.format_help().replace("usage: ", "")

    args = args_parser.parse_args()

    args.command(args)


if __name__ == '__main__':
    main()
