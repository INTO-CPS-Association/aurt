import unittest

import matplotlib.pyplot as plt
import numpy as np
from aurt.data_processing import load_data, ur5e_fields
from aurt.file_system import from_project_root
from aurt.globals import Njoints, get_ur3e_parameters, get_ur3e_PC, get_ur5e_parameters, get_ur5e_PC
from aurt.num_sym_layers import npvector, npzeros_array
from tests.utils.plotting import draw_robot, plot_dynamics
from tests import NONINTERACTIVE
from tests.utils.timed_test import TimedTest
from tests import logger


class PlottingTests(TimedTest):

    def test_plot_robot(self):

        qs = []
        q0 = np.zeros(Njoints + 1)
        # q0[2] = math.pi/2
        qs.append(q0)
        q = np.zeros(Njoints + 1)
        # q[0] = math.pi / 4
        # q[3] = -math.pi/2
        # q[4] = -math.pi/2
        qs.append(q)

        # for j in range(1, Njoints + 1):
        #     q = np.zeros(Njoints + 1)
        #     q[j] = math.pi/4
        #     qs.append(q)
        #     qs.append(q0)

        (_, d, a, alpha) = get_ur5e_parameters(npzeros_array)
        PC = get_ur5e_PC(a, npvector)
        ani = draw_robot(d, a, alpha, PC,
                         qs, 2000, repeat=True)

        if not NONINTERACTIVE:
            plt.show()
        plt.close()

    def test_animate_robot(self):

        q0 = np.zeros(Njoints + 1)
        q1 = np.ones(Njoints + 1)
        qs = [q0]  # , q1]

        (_, d, a, alpha) = get_ur5e_parameters(npzeros_array)
        logger.debug(f"d: {d}")
        logger.debug(f"a: {a}")
        logger.debug(f"alpha: {alpha}")
        PC = get_ur5e_PC(a, npvector)
        ani = draw_robot(d, a, alpha, PC, qs, 2000, repeat=True)

        if not NONINTERACTIVE:
            plt.show()
        plt.close()

    def test_plot_data(self):
        if NONINTERACTIVE:
            time_frame = (28.5, 29.5)
            step = 10
        else:
            time_frame = (-np.inf, np.inf)
            step = 10

        time_range, data = load_data(from_project_root("resources/Dataset/ur5e_all_joints_same_time/random_motion.csv"),
                                     ur5e_fields,
                                     desired_timeframe=time_frame,
                                     delimiter=' ',
                                     sample_step=step,
                                     interpolate_missing_samples=True)

        plot_dynamics(time_range, data, joints=range(1, Njoints + 1), prefix="actual")

    def test_plot_robot_animation(self):
        if NONINTERACTIVE:
            time_frame = (0, 1.0)
        else:
            time_frame = (8, np.inf)

        step = 20
        interval_ms = 1

        time_range, data = load_data(from_project_root("resources/Dataset/ur5e_all_joints_same_time/random_motion.csv"),
                                     ur5e_fields,
                                     desired_timeframe=time_frame,
                                     delimiter=' ',
                                     sample_step=step)

        # Convert loaded data into animation frames.
        qs = []
        for i in range(0, len(time_range)):
            q = np.array([data[f"target_q_{j}"][i] for j in range(0, Njoints + 1)])
            qs.append(q)

        (_, d, a, alpha) = get_ur5e_parameters(npzeros_array)
        PC = get_ur5e_PC(a, npvector)
        ani = draw_robot(d, a, alpha, PC,
                         qs, interval_ms)

        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        # ani.save('lines.mp4', writer=writer)

        if not NONINTERACTIVE:
            plt.show()
        plt.close()

    def test_draw_robot(self):
        rotate_joint = 4

        qs = []
        q0 = np.zeros(Njoints + 1)
        # q0[2] = -math.pi/2
        qs.append(q0)
        q = np.zeros(Njoints + 1)
        # q[6] = math.pi / 2
        # q[2] = -math.pi/2
        # q[4] = -math.pi/2
        qs.append(q)

        # for j in range(1, Njoints + 1):
        #     q = np.zeros(Njoints + 1)
        #     q[j] = math.pi/4
        #     qs.append(q)
        #     qs.append(q0)

        (_, d, a, alpha) = get_ur3e_parameters(npzeros_array)
        PC = get_ur3e_PC(a, npvector)
        ani = draw_robot(d, a, alpha, PC,
                         qs, 2000, repeat=True)

        if not NONINTERACTIVE:
            plt.show()
        plt.close()


if __name__ == '__main__':
    unittest.main()
