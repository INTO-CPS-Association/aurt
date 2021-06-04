import sys
import numpy as np
import sympy as sp
from scipy import signal

from aurt.calibration_aux import find_nonstatic_start_and_end_indices


class RobotCalibration:
    def __init__(self, robot_data, robot_dynamics):
        self.robot_data = robot_data
        self.robot_dynamics = robot_dynamics
        self.f_dyn = 10  # Approx. cut-off frequency of robot dynamics

    def observation_matrix(self):

        return 1

    def compute_observation_matrix_and_measurement_vector(self, regressor_base_params_instatiated):
        sys.setrecursionlimit(int(1e6))  # Prevents errors in sympy lambdify

        # Data
        qd_target = np.array([self.robot_data.data[f"target_qd_{j}"] for j in range(1, self.robot_dynamics.n_joints + 1)])
        idx_start, idx_end = find_nonstatic_start_and_end_indices(qd_target)
        q_m = np.array([self.robot_data.data[f"actual_q_{j}"] for j in
                        range(1, self.robot_dynamics.n_joints + 1)])  # (6 x n_samples) numpy array of measured angular positions
        # plot_trajectories(t, data, joints=range(2,3))

        # Low-pass filter (smoothen) measured angular position and obtain 1st and 2nd order time-derivatives
        q_tf, qd_tf, qdd_tf = RobotCalibration.__trajectory_filtering_and_central_difference(q_m,
                                                                                             self.robot_data.dt_nominal,
                                                                                             self.f_dyn,
                                                                                             idx_start,
                                                                                             idx_end)

        # *************************************************** PLOTS ***************************************************
        # qd_m = np.gradient(q_m, dt, edge_order=2, axis=1)
        # qdd_m = (q_m[:, 2:] - 2 * q_m[:, 1:-1] + q_m[:, :-2]) / (dt ** 2)  # two fewer indices than q and qd
        #
        # t = t[idx_start:idx_end]
        # qd_m = qd_m[:, idx_start:idx_end]
        # qdd_m = qdd_m[:, idx_start - 1:idx_end - 1]
        #
        # _, axs = plt.subplots(3, 1, sharex='all')
        # axs[0].set(ylabel='Position [rad]')
        # axs[1].set(ylabel='Velocity [rad/s]')
        # axs[2].set(ylabel='Acceleration [rad/s^2]')
        #
        # for j in range(Njoints):
        #     # Actual
        #     axs[0].plot(t, q_m[j,:], ':', color=plot_colors[j], label=f"actual_{j}")
        #     axs[1].plot(t, qd_m[j,:], ':', color=plot_colors[j], label=f"actual_{j}")
        #     axs[2].plot(t, qdd_m[j,:], ':', color=plot_colors[j], label=f"actual_{j}")
        #     # Filtered
        #     axs[0].plot(t, q_tf[j,:], '--', color=plot_colors[j], label=f"filtered_{j}")
        #     axs[1].plot(t, qd_tf[j,:], '--', color=plot_colors[j], label=f"filtered_{j}")
        #     axs[2].plot(t, qdd_f[j,:], '--', color=plot_colors[j], label=f"filtered_{j}")
        #     # Target
        #     axs[0].plot(t, data[f"target_q_{j+1}"][idx_start:idx_end], color=plot_colors[j], label=f"target_{j}")
        #     axs[1].plot(t, data[f"target_qd_{j+1}"][idx_start:idx_end], color=plot_colors[j], label=f"target_{j}")
        #     axs[2].plot(t, data[f"target_qdd_{j+1}"][idx_start:idx_end], color=plot_colors[j], label=f"target_{j}")
        #
        # for ax in axs:
        #     ax.legend()
        #
        # if not NONINTERACTIVE:
        #     plt.show()
        # *************************************************************************************************************

        i = np.array([self.robot_data.data[f"actual_current_{j}"] for j in range(1, self.robot_dynamics.n_joints + 1)]).T
        i_pf = RobotCalibration.__parallel_filter(i, self.robot_data.dt_nominal, self.f_dyn)[idx_start:idx_end, :]
        i_pf_ds = RobotCalibration.__downsample(i_pf, self.robot_data.dt_nominal, self.f_dyn)
        measurement_vector = i_pf_ds.flatten(order='F')  # y = [y1, ..., yi, ..., yN],  yi = [yi_{1}, ..., yi_{n_samples}]

        n_samples_ds = i_pf_ds.shape[0]  # No. of samples in downsampled data
        observation_matrix = np.zeros((self.robot_dynamics.n_joints * n_samples_ds, self.robot_dynamics.number_of_parameters))  # Initialization
        args_sym = self.robot_dynamics.q[1:] + self.robot_dynamics.qd[1:] + self.robot_dynamics.qdd[1:]  # List concatenation
        assert len(args_sym) == 3 * Njoints

        if load_model.lower() != 'none':
            args_sym += tauJ[1:]
            assert len(args_sym) == 4 * Njoints

        if load_model.lower() != 'none':
            tauJ_tf = compute_joint_torque_basis(q_tf, qd_tf, qdd_tf, robot_param_function=get_ur5e_parameters)
            args_num = np.concatenate((q_tf, qd_tf, qdd_tf, tauJ_tf[1:, :]))
        else:
            args_num = np.concatenate((q_tf, qd_tf, qdd_tf))

        for j in range(self.robot_dynamics.n_joints):  # TODO: could be computed using multiple processes
            nonzeros_j = [not elem.is_zero for elem in regressor_base_params_instatiated[j, :]]  # TODO: could be moved out of joint loop by re-writing

            # Obtain j'th row of the regressor matrix as a function of only the trajectory variables q, qd, and qdd
            # regressor_base_params_instatiated_j = sp.lambdify(args_sym,
            #                                                   regressor_base_params_instatiated[j, nonzeros_j], 'numpy')
            regressor_base_params_instatiated_j = sp.lambdify(args_sym,
                                                              self.robot_dynamics.regressor()[j, nonzeros_j], 'numpy')

            rows_j = regressor_base_params_instatiated_j(*args_num).transpose().squeeze()  # (1 x count(nonzeros))

            # Parallel filter and decimate/downsample the rows of the observation matrix related to joint j.
            rows_j_pf_ds = RobotCalibration.__downsample(RobotCalibration.__parallel_filter(rows_j, self.robot_data.dt_nominal, self.f_dyn), self.robot_data.dt_nominal, self.f_dyn)

            observation_matrix[j * n_samples_ds:(j + 1) * n_samples_ds, nonzeros_j] = rows_j_pf_ds

        return observation_matrix, measurement_vector

    @staticmethod
    def __downsample(y, dt, f_dyn):
        """The decimate procedure down-samples the signal such that the matrix system (that is later to be inverted) is not
        larger than strictly required. The signal.decimate() function can also low-pass filter the signal before
        down-sampling, but for IIR filters unfortunately only the Chebyshev filter is available which has (unwanted) ripple
        in the passband unlike the Butterworth filter that we use. The approach for downsampling is simply picking every
        downsampling_factor'th sample of the data."""

        downsampling_factor = round(0.8 / (4 * f_dyn * dt))  # downsampling_factor = 10 for dt = 0.002 s and f_dyn = 10 Hz
        y_ds = y[::downsampling_factor, :]
        return y_ds

    @staticmethod
    def __parallel_filter(y, dt, f_dyn):
        """Applies a 4th order Butterworth (IIR) filter for each row in y having a cutoff frequency of 2*f_dyn."""
        # Link to cut-off freq. eq.: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6151858
        parallel_filter_order = 4
        cutoff_freq_parallel = 2 * f_dyn
        parallel_filter = signal.butter(parallel_filter_order, cutoff_freq_parallel, btype='low', output='sos', fs=1/dt)

        y_pf = signal.sosfiltfilt(parallel_filter, y, axis=0)
        return y_pf

    @staticmethod
    def __trajectory_filtering_and_central_difference(q_m, dt, f_dyn, idx_start, idx_end):
        trajectory_filter_order = 4
        cutoff_freq_trajectory = 5 * f_dyn  # Cut-off frequency should be around 5*f_dyn = 50 Hz(?)
        trajectory_filter = signal.butter(trajectory_filter_order, cutoff_freq_trajectory, btype='low', output='sos',
                                          fs=1 / dt)
        q_tf = signal.sosfiltfilt(trajectory_filter, q_m, axis=1)

        # Obtain first and seond order time-derivatives of measured and filtered trajectory
        qd_tf = np.gradient(q_tf, dt, edge_order=2, axis=1)
        # Using the gradient function a second time to obtain the second-order time derivative would result in
        # additional unwanted smoothing, see https://stackoverflow.com/questions/23419193/second-order-gradient-in-numpy
        qdd_tf = (q_tf[:, 2:] - 2 * q_tf[:, 1:-1] + q_tf[:, :-2]) / (dt ** 2)  # two fewer indices than q and qd

        # Truncate data
        q_tf = q_tf[:, idx_start:idx_end]
        qd_tf = qd_tf[:, idx_start:idx_end]
        qdd_tf = qdd_tf[:, idx_start - 1:idx_end - 1]  # shifted due to a "lost" index in the start of the dataset

        assert q_tf.shape == qd_tf.shape == qdd_tf.shape

        return q_tf, qd_tf, qdd_tf

    def measurement_vector(self):

        return 1

    def calibrate(self):

        return 1
