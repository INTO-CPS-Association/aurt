import argparse
import traceback
from os import getcwd
import logging

from aurt.src.utils import *


def main():
    # Set up logger
    logger = logging.getLogger("aurt")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        "-fn",
        dest="filename",
        type=str,
        help="name of the file with the modified denavit hartenberg parameters"
    )
    args = parser.parse_args()

    if args.filename:
        filename = args.filename
        print(f"Argument filename: {filename}\n")
        try:
            if filename[-3:] == "csv" or filename[-4:] == "xlsx":
                d, a, alpha = convert_file_to_mdh(filename)
                # Create symbolic m
                m = [sp.symbols(f"m{i}") for i in range(len(d))]
            else:
                raise FileNotFoundError(
                    "The inputted file can either not be or found, or is not supported. Supported file formats: xlsx, csv.")
            logger.debug(f"\nm: {m}\nd: {d}\na: {a}\nalpha: {alpha}")
        except FileNotFoundError:
            traceback.print_exc()
            logger.error(f"The current directory is: {getcwd()}")

    # m0, m1, m2, m3, m4, m5, m6 = sp.symbols("m0 m1 m2 m3 m4 m5 m6")
    # m = [m0, m1, m2, m3, m4, m5, m6]
    def mdh_param_function(
            zero_array):  # TODO: make the input to the torque_algorithm_factory_from_parameters, NOT a function, but just constants instead
        return m, d, a, alpha

    calibration(mdh_param_function)


if __name__ == '__main__':
    main()
