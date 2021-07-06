
import unittest
import pickle
import os

from aurt import api

class APITests(unittest.TestCase):

    def test_compile_rbd(self):
        
        mdh_path = "aurt/tests/resources/two_link_model.csv"
        gravity = [0.0, -9.81, 0.0]
        output_path = "rbd_twolink"

        api.compile_rbd(mdh_path, gravity, output_path)

        filename = os.path.join(os.getcwd(),"cache", output_path + ".pickle")
        with open(filename, 'rb') as f:
            rbd_twolink_estimate = pickle.load(f)
        with open("aurt/tests/resources/rbd_twolink.pickle", 'rb') as f:
            rbd_twolink_true = pickle.load(f)

        self.assertEqual(rbd_twolink_estimate,rbd_twolink_true)




if __name__ == '__main__':
    unittest.main()
