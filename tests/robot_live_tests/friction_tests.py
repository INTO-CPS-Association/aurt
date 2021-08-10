import math
import time
import unittest

import numpy as np
import urinterface

from aurt.robot_data import RobotData


class FrictionTests(unittest.TestCase):

    def test_calibration_motion(self):
        filename = "friction_test_j1.csv"
        ur5e_ip = "192.168.2.40"
        sampling_rate = 500
        ur5e = urinterface.RobotConnection(ur5e_ip)
        ur5e.start_recording(filename=filename, overwrite=True, frequency=sampling_rate)
        time.sleep(0.5)
        ur5e.play_program()
        ur5e.stop_program()
        ur5e.stop_recording()

    def test_friction_shoulder(self):
        def deg2rad(x): return x*math.pi/180
        filename = "friction_test_shoulder.csv"
        ur5e_ip = "192.168.2.40"
        sampling_rate = 100
        ur5e = urinterface.RobotConnection(ur5e_ip)
        forward_angle_deg = 5
        backward_angle_deg = 2
        v_max = 2
        a_max = 0.5

        q0 = deg2rad(np.array([-90, -135, 0, -90, 180, 0]))  # Initial position
        ur5e.movej(q0, v=v_max, a=a_max)
        q1 = q0
        q2 = q0
        ur5e.start_recording(filename=filename, overwrite=True, frequency=sampling_rate)
        while deg2rad(-225) < q2[1] < deg2rad(-45):
            q2 = q1 + deg2rad(np.array([0, forward_angle_deg, 0, 0, 0, 0]))
            ur5e.movej(q2, v=v_max, a=a_max)
            q1 = q2 - deg2rad(np.array([0, backward_angle_deg, 0, 0, 0, 0]))
            ur5e.movej(q1, v=v_max, a=a_max)
        ur5e.stop_recording()

    def test_friction_elbow(self):
        def deg2rad(x): return x*math.pi/180

        filename = "friction_test_elbow.csv"
        ur5e_ip = "192.168.2.40"
        ur5e = urinterface.RobotConnection(ur5e_ip)
        forward_angle_deg = 5
        backward_angle_deg = 2
        vel = 2
        acc = 0.5

        q0 = deg2rad(np.array([-90, -135, -30, -90, 180, 0]))  # Initial position
        ur5e.movej(q0, v=vel, a=acc)
        q1 = q0
        q2 = q0
        ur5e.start_recording(filename=filename, overwrite=True, frequency=500)
        while deg2rad(-35) < q2[2] < deg2rad(115):
            q2 = q1 + deg2rad(np.array([0, 0, forward_angle_deg, 0, 0, 0]))
            ur5e.movej(q2, v=vel, a=acc)
            q1 = q2 - deg2rad(np.array([0, 0, backward_angle_deg, 0, 0, 0]))
            ur5e.movej(q1, v=vel, a=acc)
        ur5e.stop_recording()

    def test_friction_plot(self):
        filename_shoulder = "friction_test_shoulder.csv"
        # filename_elbow = "friction_test_elbow.csv"
        rd_shoulder = RobotData(filename_shoulder, delimiter=' ', interpolate_missing_samples=True)
        # rd_elbow = RobotData(filename_elbow, delimiter=' ')
        rd_shoulder.plot_missing_samples()
        # rd_elbow.plot_sampling()

    def test_calibration_plot(self):
        filename_shoulder = "calibration_motion.csv"
        # filename_elbow = "friction_test_elbow.csv"
        rd_shoulder = RobotData(filename_shoulder, delimiter=' ', interpolate_missing_samples=True)
        # rd_elbow = RobotData(filename_elbow, delimiter=' ')
        rd_shoulder.plot_missing_samples()
        # rd_elbow.plot_sampling()

if __name__ == '__main__':
    unittest.main()
