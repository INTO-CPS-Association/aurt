# File for running:
#          - tests
#          - live tests
# Please run this file and make sure the tests all pass before pushing any content.
import argparse
import unittest
import pathlib

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-tests",
        dest="run_tests",
        choices=["api","full-offline","live"],
        help="when set, the dynamic calibration tests will be run, depending on the choice: 'api','full-offline','live'. Live tests can be performed on a UR robot."
    )
    args = parser.parse_args()

    if args.run_tests == "full-offline":
        suite = unittest.TestSuite()
        suite.addTest(unittest.TestLoader().discover("tests", pattern="*tests.py"))
        runner = unittest.TextTestRunner()
        runner.run(suite)
    elif args.run_tests == "api":
        suite = unittest.TestSuite()
        suite.addTest(unittest.TestLoader().discover("aurt/tests", pattern="*tests.py"))
        runner = unittest.TextTestRunner()
        runner.run(suite)
    elif args.run_tests == "live":
        try:
            loader = unittest.TestLoader()
            start_dir = pathlib.Path('robot_live_tests')
            suite = loader.discover(start_dir, pattern="*tests.py")
            runner = unittest.TextTestRunner()
            runner.run(suite)
        except:
            pass