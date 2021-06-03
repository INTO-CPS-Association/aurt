import unittest

from aurt.data_processing import load_data, ur5e_fields
from aurt.file_system import from_project_root
from tests import NONINTERACTIVE

import os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import aurt.calibration_aux as cal_aux
from tests.utils.timed_test import TimedTest


def select_data_subset(data, idx_start, idx_end):
    data_result = {}
    for f in data:
        data_result[f] = data[f][idx_start:idx_end]
    assert len(data_result.keys()) == len(data.keys())
    return data_result


class FrictionTests(TimedTest):

    def test_constant_velocity_no_load(self):
        path = 'resources/Dataset/ur5e_constant_velocity_no_load'

        v = []  # velocity
        f = []  # friction
        sigma_v = []  # standard deviation of velocity
        sigma_f = []  # standard deviation of friction
        conf_int_v = []  # confidence interval of velocity
        conf_int_f = []  # confidence interval of friction

        i = 0
        with os.scandir(from_project_root(path)) as dir_iterator:
            csv_files = [file for file in dir_iterator if file.name.endswith("csv")]
            for entry in csv_files:
                print(entry.path)

                time_range, data = load_data(entry.path, ur5e_fields)

                idx_start, idx_end = cal_aux.find_const_vel_start_and_end_indices(time_range, data["target_qd_1"])
                subset_data = select_data_subset(data, idx_start, idx_end)
                len_data = idx_end - idx_start

                v.append(subset_data["actual_qd_1"].mean())
                f.append(subset_data["actual_current_1"].mean())
                sigma_v.append(np.std(subset_data["actual_qd_1"]))
                sigma_f.append(np.std(subset_data["actual_current_1"]))

                conf_int_v.append(stats.norm.interval(0.95, loc=f[-1], scale=sigma_v[-1] / np.sqrt(len_data))
                                  [1])  # 95 % confidence interval on v
                conf_int_f.append(stats.norm.interval(0.95, loc=f[-1], scale=sigma_f[-1] / np.sqrt(len_data))
                                  [1])  # 95 % confidence interval on f

                """
                fig, axs = plt.subplots(2, 1, sharex=True)
                axs[0].plot(time_range, data["actual_qd_1"], label='actual_q_1')
                axs[0].plot([time_range[idx_start], time_range[idx_end]], [v[-1], v[-1]], '--',
                            label='average')
                axs[0].plot([time_range[idx_start], time_range[idx_end]],
                            [v[-1] - sigma_v[-1], v[-1] - sigma_v[-1]], 'k--', label='standard deviation')
                axs[0].plot([time_range[idx_start], time_range[idx_end]],
                            [v[-1] + sigma_v[-1], v[-1] + sigma_v[-1]], 'k--')
                axs[0].legend()
                axs[0].set(ylabel='Angular Velocity [rad/s]')
                axs[1].plot(time_range, data["actual_current_1"], label='actual_current_1')
                axs[1].plot([time_range[idx_start], time_range[idx_end]], [f[-1], f[-1]], '--',
                            label='average')
                axs[1].plot([time_range[idx_start], time_range[idx_end]],
                            [f[-1] - sigma_f[-1], f[-1] - sigma_f[-1]], 'k--', label='standard deviation')
                axs[1].plot([time_range[idx_start], time_range[idx_end]],
                            [f[-1] + sigma_f[-1], f[-1] + sigma_f[-1]], 'k--')
                axs[1].set(xlabel='Time [s]')
                axs[1].set(ylabel='Current [A]')
                axs[1].legend()
                plt.xlim((0, time_range[-1]))


            if not NONINTERACTIVE:
                plt.show()
                """

        v_neg = cal_aux.negate_negative(v)
        f_neg = cal_aux.negate_negative(f)
        p1 = plt.plot(v, f, marker='x', linestyle='none', label='original data')
        color_orig = p1[0].get_color()
        plt.plot(v_neg, f_neg, marker='o', linestyle='none', fillstyle='none',
                 label='negated negative-velocity data')
        lineStyle_orig = {"linestyle": "none", "capsize": 3}
        plt.errorbar(v, f, xerr=sigma_v, yerr=sigma_f, **lineStyle_orig, color=color_orig, label='standard deviation')
        #plt.errorbar(v, f, xerr=conf_int_v, yerr=conf_int_f, **lineStyle_orig, color=color_orig, label='95 % confidence interval')
        plt.grid(which='major')
        plt.xlabel('Angular Velocity [rad/s]')
        plt.ylabel('Friction [A]')
        plt.legend()

        if not NONINTERACTIVE:
            plt.show()


if __name__ == '__main__':
    unittest.main()
