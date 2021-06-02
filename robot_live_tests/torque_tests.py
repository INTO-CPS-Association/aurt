import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
from urinterface import RobotConnection

from aurt.data_processing import load_data, ur5e_fields
from aurt import Njoints, g, get_ur5e_parameters, get_ur5e_PC
from aurt import draw_robot
from aurt import compute_torques_numeric_5e, npzeros_array, npvector
from tests import NONINTERACTIVE
from tests.utils.timed_test import TimedTest


class TorqueTests(TimedTest):

    def test_numeric_torques_q_single_sample(self):
        ur5e = RobotConnection("192.168.56.101")
        file_name = 'test_numeric_torques_single_sample.csv'
        q = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 0.0])
        qd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        qdd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        f_tip = np.zeros((3, 1))
        n_tip = np.zeros((3, 1))
        inertia = [np.zeros((3, 3)) for j in range(Njoints + 1)]

        ur5e.movej(q=q)
        ur5e.record_samples(filename=file_name, overwrite=True, samples=1)
        tau_ours = compute_torques_numeric_5e(q=np.insert(q, 0, 0.0), der_q=np.insert(qd, 0, 0.0), der2_q=np.insert(qdd, 0, 0.0), f_tip=f_tip, n_tip=n_tip, cI=inertia, g=g)

        data = load_data(file_name,
                         ur5e_fields,
                         delimiter=' ')[1]

        q_UR = np.array([data[f"target_q_{j}"][0] for j in range(1, Njoints + 1)])
        tau_UR = np.array([data[f"target_moment_{j}"][0] for j in range(1, Njoints + 1)])
        np.set_printoptions(precision=3, suppress=True)

        print(f"q_ours = {q}")
        print(f"q_UR = {q_UR}\n")
        print(f"tau_UR   = {tau_UR}")
        print(f"tau_ours   = {tau_ours}")
        assert (np.linalg.norm(q - q_UR) < 1.0E-3)

        # Does the robot go to the same position?  q == test_numeric_data.csv.q()
        # What's the torque?  tau_num == test_numeric_data.csv.tau()

        (_, d, a, alpha) = get_ur5e_parameters(npzeros_array)
        PC = get_ur5e_PC(a, npvector)
        ani = draw_robot(d, a, alpha, PC,
                         [np.insert(q, 0, 0.0)], 2000, repeat=True)

        if not NONINTERACTIVE:
            plt.show()
        plt.close()

    def test_generate_test_data(self):
        ur5e = RobotConnection("192.168.56.101")
        file_name = 'test_numeric_torques_multiple_samples.csv'
        q0 = np.array([0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0])
        q1 = np.array([1.57, 0.0, -1.5708, 0, 3.1415, 3.1415])

        ur5e.movej(q=q0)
        ur5e.start_recording(filename=file_name, overwrite=True)
        time.sleep(1.5)
        ur5e.movej(q=q1)
        time.sleep(1.0)
        ur5e.stop_recording()


if __name__ == '__main__':
    unittest.main()
