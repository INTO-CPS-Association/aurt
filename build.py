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
        choices=["api","cli","slow_tests","live"],
        help="when set, the dynamic calibration tests will be run, depending on the choice: 'api','cli','slow_tests','live'. Live tests can be performed on a UR robot."
    )
    args = parser.parse_args()

    if args.run_tests == "slow_tests":
        suite = unittest.TestSuite()
        suite.addTest(unittest.TestLoader().discover("tests/slow_tests", pattern="*tests.py"))
        runner = unittest.TextTestRunner()
        res = runner.run(suite)
    elif args.run_tests == "api":
        suite = unittest.TestSuite()
        suite.addTest(unittest.TestLoader().discover("aurt/tests", pattern="api_tests.py"))
        runner = unittest.TextTestRunner()
        res = runner.run(suite)
    elif args.run_tests ==  "cli":
        suite = unittest.TestSuite()
        suite.addTest(unittest.TestLoader().discover("aurt/tests", pattern="cli_tests.py"))
        runner = unittest.TextTestRunner()
        res = runner.run(suite)
    elif args.run_tests == "live":
        try:
            loader = unittest.TestLoader()
            start_dir = pathlib.Path('tests/robot_live_tests')
            suite = loader.discover(start_dir, pattern="*tests.py")
            runner = unittest.TextTestRunner()
            runner.run(suite)
        except:
            pass
    
    if res.wasSuccessful():
        exit(0)
    else:
        exit(1)