import math
import time
import unittest
import numpy as np
from urinterface import RobotConnection


class RobotLiveTests(unittest.TestCase):

    def test_start_stop_recording_ur5e(self):
        ur5e = RobotConnection("192.168.12.245")  # UR5e Robot @ DeepTech
        ur5e.start_recording(filename=f'randomMotion2.csv')
        ur5e.stop_recording()

    def test_const_vel_no_load(self):
        # ur5e = RobotConnection("192.168.56.101") # VirtualBox
        ur5e = RobotConnection("192.168.12.245")  # UR5e robot @ DeepTech
        q0 = np.array([-2. * math.pi, 0., -2.7925, -.5 * math.pi, 0., 0.])
        q1 = np.array([2. * math.pi, 0., -2.7925, -.5 * math.pi, 0., 0.])
        acc = 20.0
        v_abs_array = np.linspace(0.2, math.pi, 20)

        ur5e.movej(q=q1)  # move to start
        i = 1
        for v_abs in v_abs_array:
            # record negative velocity
            ur5e.start_recording(filename=f'velTest3_{i}_v=-{v_abs:.4f}.csv')
            time.sleep(1.5)  # wait for thread to start recording
            ur5e.movej(q=q0, v=v_abs, a=acc)
            time.sleep(1.5)  # wait for thread to start recording
            ur5e.stop_recording()

            # record positive velocity
            ur5e.start_recording(filename=f'velTest3_{i}_v={v_abs:.4f}.csv')
            time.sleep(1.5)
            ur5e.movej(q=q1, v=v_abs, a=acc)
            time.sleep(1.5)
            ur5e.stop_recording()

            i += 1


    def test_const_vel_with_load(self):
        ur5e = RobotConnection("192.168.12.245")  # Robot @ DeepTech
        # Shoulder angles are chosen such that the UR5e robot will just accurately not hit the MIR robot
        q0 = np.array([0., -3.57, 0., -.5*math.pi, 0., 0.])
        q1 = np.array([0., 0.837, 0., -.5*math.pi, 0., 0.])
        acc = 20.0
        vel = 0.3

        # record positive velocity
        ur5e.movej(q=q0)  # move to start
        ur5e.start_recording(filename=f'velLoad_v={vel:.4f}.csv')
        time.sleep(1.5)  # wait for thread to start recording
        ur5e.movej(q=q1, v=vel, a=acc)
        time.sleep(1.5)  # wait for thread to start recording
        ur5e.stop_recording()

        # record negative velocity
        ur5e.start_recording(filename=f'velLoad_v=-{vel:.4f}.csv')
        time.sleep(1.5)  # wait for thread to start recording
        ur5e.movej(q=q0, v=vel, a=acc)
        time.sleep(1.5)  # wait for thread to start recording
        ur5e.stop_recording()


    def test_const_slow_vel_no_load(self):
        ur5e = RobotConnection("192.168.12.245")  # Robot @ DeepTech
        q0 = np.array([0, 0., -2.7925, -.5*math.pi, 0., 0.])
        q1 = np.array([0.5*math.pi, 0., -2.7925, -.5*math.pi, 0., 0.])
        acc = 20.0
        v_abs_array = np.linspace(0.005, 0.05, 10)

        ur5e.movej(q=q1)  # move to start
        i = 1
        for v_abs in v_abs_array:
            # record negative velocity
            ur5e.start_recording(filename=f'vel_slow2_{i}_v=-{v_abs:.4f}.csv')
            time.sleep(1.5)  # wait for thread to start recording
            ur5e.movej(q=q0, v=v_abs, a=acc)
            time.sleep(1.5)  # wait for thread to start recording
            ur5e.stop_recording()

            # record positive velocity
            ur5e.start_recording(filename=f'vel_slow2_{i}_v={v_abs:.4f}.csv')
            time.sleep(1.5)
            ur5e.movej(q=q1, v=v_abs, a=acc)
            time.sleep(1.5)
            ur5e.stop_recording()

            i += 1

    def test_real_record(self):
        ur5e = RobotConnection("192.168.56.101")
        ur5e.record_samples(config_file="resources/record_configuration.xml", frequency=100, samples=10)


if __name__ == '__main__':
    unittest.main()
