import unittest

from aurt.src.data_processing import process_missing_samples
from aurt.src.file_system import safe_open
from aurt.tests.timed_test import TimedTest
import numpy as np


class DataProcessingTests(TimedTest):

    def test_process_missing_samples_no_missing_sample(self):
        time_range = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        data = {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0]
        }
        interpolate_missing_samples = True
        rtime_range, rdata = process_missing_samples(time_range, data, interpolate_missing_samples)
        self.assertTrue(not any(rdata["interpolated"]))
        self.assertTrue(np.array_equal(time_range, rtime_range))
        self.assertTrue(np.array_equal(data["x"], rdata["x"]))

    def test_process_missing_samples_missing_2samples(self):
        time_range = np.array([0.0, 1.0, 4.0, 5.0])
        data = {
            "x": [0.0, 1.0, 4.0, 5.0]
        }
        interpolate_missing_samples = True
        rtime_range, rdata = process_missing_samples(time_range, data, interpolate_missing_samples)
        self.assertTrue(any(rdata["interpolated"]))
        self.assertTrue(np.array_equal(np.array([0., 1., 2., 3., 4., 5.]), rtime_range))
        self.assertTrue(np.array_equal(np.array([0., 1., 2., 3., 4., 5.]), rdata["x"]))
        self.assertTrue(np.array_equal(np.array([False, False,  True,  True, False, False]), rdata["interpolated"]))

    def test_find_subset_data(self):
        path = './resources/Dataset/ConstantVelocityWithLoadTorque'

        v = []
        f = []
        sigma = []
        n_samples_stabilize = 200

        filename = 'velLoadTest.csv'
        with safe_open(filename) as csvfile:
            rd = csv_reader.CSVReader(csvfile)  # robot data object

        # find start and end indices for useful subset of data:
        # 1. Find center of time series data
        # 1a. Check if qdd_target(signal center) == 0 (constant angular velocity is assumed)
        #     From experience, the qdd_target provided by the UR controller is erroneous, thus we use:
        #          qdd_target_new = diff(qd_target)
        # 2. Find first nonzero qdd_target_new in each direction.
        # 3. For idx_start = idx_start + fixed_number (for instance 200 = 0.4 second with 500 Hz sampling) due to
        #    non-steady state/oscillations due to ur3e_acceleration change.

        i = 0
        for entry in os.scandir(path):
            print(entry.path)
            with safe_open(entry.path) as csvfile:
                rd = ase.CSVReader(csvfile)  # robot data object

            # find start and end indices for useful subset of data:
            # 1. Find center of time series data
            # 1a. Check if qdd_target(signal center) == 0 (constant angular velocity is assumed)
            #     From experience, the qdd_target provided by the UR controller is erroneous, thus we use:
            #          qdd_target_new = diff(qd_target)
            # 2. Find first nonzero qdd_target_new in each direction.
            # 3. For idx_start = idx_start + fixed_number (for instance 200 = 0.4 second with 500 Hz sampling) due to
            #    non-steady state/oscillations due to ur3e_acceleration change.

            qdd_threshold = 0.1
            idx_mid = rd.timestamp.size // 2
            dx = rd.timestamp[1] - rd.timestamp[0]
            qdd_target_new = np.diff(rd.target_qd_0, prepend=0.0) / dx
            # TODO: If not qdd_target_new(idx_mid) == 0, analyse entire qdd_target_new and find the longest duration of
            #  qdd_target_new == 0 while (in addition) qd_target != 0.
            assert (abs(qdd_target_new[idx_mid]) < qdd_threshold)
            idx_start = n_samples_stabilize + idx_mid - find_first(
                (-qdd_threshold > qdd_target_new[0:idx_mid][::-1]) | (
                        qdd_target_new[0:idx_mid][
                        ::-1] > qdd_threshold))
            idx_end = idx_mid - 2 + find_first(
                (-qdd_threshold > qdd_target_new[idx_mid:-1]) | (qdd_target_new[idx_mid:-1] > qdd_threshold))

            v.append(rd.actual_qd_0[idx_start:idx_end].mean())
            f.append(rd.actual_current_0[idx_start:idx_end].mean())
            sigma.append(np.std(rd.actual_current_0[idx_start:idx_end]))

            """
            t0 = rd.timestamp[0]
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].plot(rd.timestamp - t0, rd.actual_qd_0, label='actual_q_0')
            axs[0].plot([rd.timestamp[idx_start] - t0, rd.timestamp[idx_end] - t0], [v[-1], v[-1]], '--', label='average')
            axs[0].legend()
            axs[0].set(ylabel='Angular Velocity [rad/s]')
            axs[1].plot(rd.timestamp - t0, rd.actual_current_0, label='actual_current_0')
            axs[1].plot([rd.timestamp[idx_start] - t0, rd.timestamp[idx_end] - t0], [f[-1], f[-1]], '--', label='average')
            axs[1].set(xlabel='Time [s]')
            axs[1].set(ylabel='Current [A]')
            axs[1].legend()
            plt.xlim((0, rd.timestamp[-1] - t0))
            plt.show()
            """

            i += 1

        def neg(lst):  # return negated value of negative elements
            return [-x for x in lst if x < 0.0] or None

        v_neg = neg(v)
        f_neg = neg(f)
        p1 = plt.plot(v, f, marker='x', linestyle='none', label='original data')
        print(p1[0].get_color())
        color_orig = p1[0].get_color()
        plt.plot(v_neg, f_neg, marker='o', linestyle='none', fillstyle='none', label='negated negative-velocity data')
        lineStyle_orig = {"linestyle": "none",
                          "capsize": 3}  # {"linestyle":"--", "linewidth":2, "markeredgewidth":2, "elinewidth":2, "capsize":3}
        plt.errorbar(v, f, yerr=sigma, **lineStyle_orig, color=color_orig, label='standard deviation')
        plt.grid(which='major')
        plt.xlabel('Angular Velocity [rad/s]')
        plt.ylabel('Friction [A]')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    unittest.main()
