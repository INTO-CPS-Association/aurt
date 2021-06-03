import unittest

from aurt import api


class APITests(unittest.TestCase):

    def test_compile_rbd(self):
        # TODO: Complete test
        mdh_path = ""
        gravity = [0.0, 0.0, -9.81]
        output_path = ""

        api.compile_rbd(mdh_path, gravity, output_path)
