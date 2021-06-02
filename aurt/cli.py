import argparse
import traceback
from os import getcwd
import logging

from aurt.data_processing import convert_file_to_mdh


def main():
    # Set up logger
    logger = logging.getLogger("aurt")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser()

    parser.add_argument('command', choices=['info', 'validate', 'simulate', 'compile', 'add-cswrapper', 'add-remoting',
                                            'create-cmake-project', 'create-jupyter-notebook'],
                        help="Command to execute")


    args = parser.parse_args()

    if args.filename:
        filename = args.filename
        print(f"Argument filename: {filename}\n")
        try:
            if filename[-3:] == "csv":
                d, a, alpha = convert_file_to_mdh(filename)
            else:
                raise FileNotFoundError(
                    "The inputted file can either not be or found, or is not supported. Supported file formats: csv.")
            logger.debug(f"\nm: {m}\nd: {d}\na: {a}\nalpha: {alpha}")
        except FileNotFoundError:
            traceback.print_exc()
            logger.error(f"The current directory is: {getcwd()}")


if __name__ == '__main__':
    main()
