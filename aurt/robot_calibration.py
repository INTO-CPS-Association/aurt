import multiprocessing
import numpy as np
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from logging import Logger
from multiprocessing import Pool

from aurt.robot_dynamics import RobotDynamics
from aurt.signal_processing import central_finite_difference
from aurt.calibration_aux import find_nonstatic_start_and_end_indices
from aurt.robot_data import RobotData, plot_colors


class RobotCalibration:
    f_dyn = 10  # Approximate cut-off frequency [Hz] of robot dynamics
    qd_tf_noise_threshold = 0.03  # Threshold [rad/s] to determine non-stationary (any(abs(qd) > qd_tf_noise_threshold)) data

    def __init__(self, l: Logger, robot_dynamics: RobotDynamics, robot_data_path, gravity, relative_separation_of_calibration_and_prediction=None,
                 robot_data_predict=None, multi_processing: bool=True):

        self.logger = l
        self.gravity = gravity
        self.robot_dynamics: RobotDynamics = robot_dynamics

        if relative_separation_of_calibration_and_prediction is None and robot_data_predict is None:
            """All data is used for calibration."""
            self.robot_data_calibration = RobotData(l, robot_data_path, delimiter=' ', interpolate_missing_samples=True)
            self.robot_data_validation = None
        elif relative_separation_of_calibration_and_prediction is not None and robot_data_predict is None:
            """Using a single dataset, some of the data is used for calibration and some is used for validation."""
            assert 0 < relative_separation_of_calibration_and_prediction < 1, "The specified relative separation of " \
                                                                              "data used for calibration and " \
                                                                              "prediction must be in the range from 0 " \
                                                                              "to 1."
            dummy_data = RobotData(l, robot_data_path, delimiter=' ')
            t_sep = dummy_data.time[-1] * relative_separation_of_calibration_and_prediction
            
            if multi_processing:
                robot_data_args = [[l, robot_data_path, (0, t_sep)],
                                   [l, robot_data_path, (t_sep, np.inf)]]
                with Pool() as p:  # Compute using multiple processes
                    robot_data_touple = p.map(self._load_robot_data_parallel, robot_data_args)
                self.robot_data_calibration, self.robot_data_validation = robot_data_touple
            else:
                self.robot_data_calibration = RobotData(l, robot_data_path, delimiter=' ', interpolate_missing_samples=True, desired_timeframe=(0, t_sep))
                self.robot_data_validation = RobotData(l, robot_data_path, delimiter=' ', interpolate_missing_samples=True, desired_timeframe=(t_sep, np.inf))
        elif relative_separation_of_calibration_and_prediction is None and robot_data_predict is not None:
            """Using two datasets, one for calibration and one for validation."""
            self.robot_data_calibration = RobotData(l, robot_data_path, delimiter=' ', interpolate_missing_samples=True)
            self.robot_data_validation = robot_data_predict
        else:
            # exit here? call __del__()?
            self.logger.error("A wrong combination of arguments was provided.")
        
        q_m = np.array([self.robot_data_calibration.data[f"actual_q_{j}"] for j in range(1, self.robot_dynamics.n_joints + 1)])
        _, qd_tf, _ = self._trajectory_filtering_and_central_difference(q_m, self.robot_data_calibration.dt_nominal, RobotCalibration.f_dyn)
        self.non_static_start_idx, self.non_static_end_idx = find_nonstatic_start_and_end_indices(qd_tf, qd_threshold=RobotCalibration.qd_tf_noise_threshold)
        self.downsampling_factor = round(0.8 / (4 * RobotCalibration.f_dyn * self.robot_data_calibration.dt_nominal))
        self.parameters = None
        self.estimated_output = None
        self.number_of_samples_in_downsampled_data = None
    
    def _load_robot_data_parallel(self, args):
        l = args[0]
        robot_data_path = args[1]
        timeframe = args[2]

        return RobotData(l, robot_data_path, delimiter=' ', interpolate_missing_samples=True, desired_timeframe=timeframe)

    def _measurement_vector(self, robot_data, start_index=None, end_index=None):
        def compute_measurement_vector():
            i = np.array([robot_data.data[f"actual_current_{j}"] for j in range(1, self.robot_dynamics.n_joints + 1)]).T
            i_pf = RobotCalibration._parallel_filter(i, robot_data.dt_nominal, RobotCalibration.f_dyn)[start_index:end_index, :]
            i_pf_ds = RobotCalibration._downsample(i_pf, self.downsampling_factor)
            return i_pf_ds.flatten(order='F')  # y = [y1, ..., yi, ..., yN],  yi = [yi_{1}, ..., yi_{n_samples}]

        return compute_measurement_vector()

    def _observation_matrix(self, robot_data, gravity, start_index=None, end_index=None):
        q_m = np.array([robot_data.data[f"actual_q_{j}"] for j in range(1, self.robot_dynamics.n_joints + 1)])

        # Low-pass filter (smoothen) measured angular position(s) and obtain 1st and 2nd order time-derivatives
        q_tf, qd_tf, qdd_tf = RobotCalibration._trajectory_filtering_and_central_difference(q_m, 
                                                                                            robot_data.dt_nominal,
                                                                                            RobotCalibration.f_dyn,
                                                                                            start_index,
                                                                                            end_index)

        if not all(gravity == self.robot_dynamics.rigid_body_dynamics._g_num):
            self.robot_dynamics.rigid_body_dynamics.instantiate_gravity(gravity)
            self.robot_dynamics.rigid_body_dynamics.name += f"_gravity={gravity}"
            self.robot_dynamics.compute_linearly_independent_system()

        n_samples_ds = self._measurement_vector(robot_data, start_index=start_index, end_index=end_index).shape[
                           0] // self.robot_dynamics.n_joints  # No. of samples in downsampled data

        observation_matrix = np.zeros((self.robot_dynamics.n_joints * n_samples_ds,
                                       sum(self.robot_dynamics.number_of_parameters())))  # Initialization
        states_num = np.empty((q_tf.shape[0] + qd_tf.shape[0] + qdd_tf.shape[0], q_tf.shape[1]))
        states_num[0::3, :] = q_tf
        states_num[1::3, :] = qd_tf
        states_num[2::3, :] = qdd_tf
        for j in range(self.robot_dynamics.n_joints):
            # Obtain the rows of the observation matrix related to joint j
            obs_mat_j = self.robot_dynamics.observation_matrix_joint(j, states_num)

            # Parallel filter and decimate/downsample the rows of the observation matrix related to joint j.
            obs_mat_j_ds = RobotCalibration._downsample(
                RobotCalibration._parallel_filter(obs_mat_j, robot_data.dt_nominal, RobotCalibration.f_dyn),
                self.downsampling_factor)
            
            observation_matrix[j * n_samples_ds:(j + 1) * n_samples_ds, :] = obs_mat_j_ds
        return observation_matrix

    def calibrate(self):
        """
        Calibrates the 'robot_dynamics' model using dataset 'robot_data_calibration' and output values of the
        calibrated parameters to file 'filename_parameters'.
        """

        observation_matrix = self._observation_matrix(self.robot_data_calibration,
                                                      gravity=self.gravity,
                                                      start_index=self.non_static_start_idx,
                                                      end_index=self.non_static_end_idx)
        measurement_vector = self._measurement_vector(self.robot_data_calibration,
                                                      start_index=self.non_static_start_idx,
                                                      end_index=self.non_static_end_idx)

        # sklearn fit
        OLS = LinearRegression(fit_intercept=False)
        OLS.fit(observation_matrix, measurement_vector)
        measurement_vector_ols_estimation = OLS.predict(observation_matrix)
        n_samples_est = round(len(measurement_vector) / self.robot_dynamics.n_joints)

        # Reshape measurement vector from (n_joints*n_samples x 1) to (n_samples x n_joints)
        assert n_samples_est * self.robot_dynamics.n_joints == len(measurement_vector_ols_estimation) \
               == np.shape(observation_matrix)[0]
        measurement_vector_reshape = np.reshape(measurement_vector, (self.robot_dynamics.n_joints, n_samples_est))
        measurement_vector_ols_estimation_reshape = np.reshape(measurement_vector_ols_estimation,
                                                               (self.robot_dynamics.n_joints, n_samples_est))

        # Compute weights (the reciprocal of the estimated variance of the error)
        residuals = measurement_vector_reshape - measurement_vector_ols_estimation_reshape
        residual_sum_of_squares = np.sum(np.square(residuals), axis=1)
        variance_residual = residual_sum_of_squares / (
                    n_samples_est - np.sum(self.robot_dynamics.number_of_parameters()))

        standard_deviation = np.sqrt(variance_residual)
        wls_sample_weights = np.repeat(1 / standard_deviation, n_samples_est)

        # Weighted Least Squares solution
        wls_calibration = LinearRegression(fit_intercept=False)
        wls_calibration.fit(observation_matrix, measurement_vector, sample_weight=wls_sample_weights)
        y_pred = wls_calibration.predict(observation_matrix)

        y_pred_reshaped = np.reshape(y_pred, (self.robot_dynamics.n_joints, y_pred.shape[0] // self.robot_dynamics.n_joints))
        measurement_vector_reshaped = np.reshape(measurement_vector, (self.robot_dynamics.n_joints, y_pred.shape[0] // self.robot_dynamics.n_joints))
        assert y_pred_reshaped.shape == measurement_vector_reshaped.shape
        mse = RobotCalibration.get_mse(measurement_vector_reshaped, y_pred_reshaped)
        normalization = np.mean(abs(measurement_vector_reshaped), axis=1)
        nmse = mse / normalization
        self.logger.info(f"MSE (calibration data): {mse}")
        self.logger.info(f"NMSE (calibration data): {nmse}")
        
        weighted_observation_matrix = (wls_sample_weights*observation_matrix.T).T
        std_dev_parameter_estimate = np.sqrt(np.diagonal(np.linalg.inv(weighted_observation_matrix.T @ weighted_observation_matrix)))  # calculates the standard deviation of the parameter estimates from the diagonal elements (variance) of the covariance matrix
        cond = RobotCalibration._evaluate_dynamics_excitation_as_cost((wls_sample_weights*observation_matrix.T).T, metric="cond")
        self.logger.info(f"Condition no. of weighted observation matrix: {cond}.")

        self.parameters = wls_calibration.coef_
        rel_std_dev_parameter_estimate = 100 * (std_dev_parameter_estimate / self.parameters)
        self.logger.info(f"Full Parameters: {self.robot_dynamics._parameters_full()}")
        self.logger.info(f"Base Parameters: {self.robot_dynamics.parameters}")
        self.logger.info(f"Relative std.dev. of estimated parameters: {rel_std_dev_parameter_estimate}")

        return self.parameters

    def predict(self, robot_data_predict: RobotData, gravity, parameters):
        observation_matrix = self._observation_matrix(robot_data_predict, gravity)
        estimated_output = observation_matrix @ parameters

        n_samples_ds = round(observation_matrix.shape[0] / self.robot_dynamics.n_joints)
        assert n_samples_ds * self.robot_dynamics.n_joints == observation_matrix.shape[0]
        output_predicted_reshaped = np.reshape(estimated_output, (self.robot_dynamics.n_joints, n_samples_ds))

        _, y_meas_reshaped, y_pred_reshaped = self._get_plot_values_for(robot_data_predict, gravity, parameters)
        mse = RobotCalibration.get_mse(y_meas_reshaped, y_pred_reshaped)
        normalization = np.mean(abs(y_meas_reshaped), axis=1)
        nmse = mse / normalization
        self.logger.info(f"MSE (validation data): {mse}")
        self.logger.info(f"NMSE (validation data): {nmse}")

        # return robot_data_predict.time[
        #       ::self.downsampling_factor], output_predicted_reshaped  # TODO: CORRECT ERROR; 'time' does not correspond in length to 'output_predicted_reshaped'
        return output_predicted_reshaped

    def _get_plot_values_for(self, data, gravity, parameters):
        observation_matrix = self._observation_matrix(data, gravity)
        n_samples = observation_matrix.shape[0] // self.robot_dynamics.n_joints
        estimated_data = np.reshape(observation_matrix @ parameters, (self.robot_dynamics.n_joints, n_samples))
        measured_data = np.reshape(self._measurement_vector(data),
                                              (self.robot_dynamics.n_joints, n_samples))
        #error = measured_data - estimated_data
        t_data = np.linspace(0, data.dt_nominal * self.downsampling_factor * n_samples, n_samples)
        return t_data, measured_data, estimated_data

    def plot_velocity(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings
            warnings.warn("The matplotlib package is not installed, please install it for plotting the calibration.")

        t = self.robot_data_calibration.data["timestamp"]
        n_joints = self.robot_dynamics.n_joints
        q_m = np.array([self.robot_data_calibration.data[f"actual_q_{j+1}"] for j in range(n_joints)])
        # _, qd_m = central_finite_difference(q_m, self.robot_data_calibration.dt_nominal, order=1)
        _, qd_tf, _ = self._trajectory_filtering_and_central_difference(q_m, self.robot_data_calibration.dt_nominal, self.f_dyn)
        qd_tf = np.insert(qd_tf, qd_tf.shape[1], np.repeat(0, n_joints, axis=0), axis=1)
        qd_tf = np.insert(qd_tf, 0, np.repeat(0, n_joints, axis=0), axis=1)

        for j in range(n_joints):
            plt.plot(t, qd_tf[j, :], color=plot_colors[j], label=f"actual_q{j}_filtered")
            # plt.plot(t, self.robot_data_calibration.data[f"target_qd_{j+1}"], color=plot_colors[j], label=f"target_q{j}")
        
        # plot non-static indices
        ymin, ymax = plt.gca().get_ylim()
        plt.plot([t[self.non_static_start_idx], t[self.non_static_start_idx]], [ymin, ymax], '--', color='black', linewidth=3, label='non_static_idx')
        plt.plot([t[self.non_static_end_idx], t[self.non_static_end_idx]], [ymin, ymax], '--', color='black', linewidth=3)
        plt.plot([t[0], t[-1]], [self.qd_tf_noise_threshold, self.qd_tf_noise_threshold], '--', color='red', linewidth=3, label='qd_noise_threshold')
        plt.plot([t[0], t[-1]], [-self.qd_tf_noise_threshold, -self.qd_tf_noise_threshold], '--', color='red', linewidth=3)

        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [rad/s]')
        plt.legend()
        plt.show()

    def plot_calibration(self, parameters):
        try:
            import matplotlib.colors
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings
            warnings.warn("The matplotlib package is not installed, please install it for plotting the calibration.")

        t, measured_output_reshaped, estimated_output_reshaped = self._get_plot_values_for(self.robot_data_calibration, self.gravity, parameters)
        error = measured_output_reshaped - estimated_output_reshaped

        fig = plt.figure()
        gs = fig.add_gridspec(2, 1, hspace=0.03)
        axs = gs.subplots(sharex='col', sharey='all')

        def darken_color(plot_color, darken_amount=0.35):
            """Computes rgb values to a darkened color"""
            line_color_rgb = matplotlib.colors.ColorConverter.to_rgb(plot_color)
            line_color_hsv = matplotlib.colors.rgb_to_hsv(line_color_rgb)
            darkened_line_color_hsv = line_color_hsv - np.array([0, 0, darken_amount])
            darkened_line_color_rgb = matplotlib.colors.hsv_to_rgb(darkened_line_color_hsv)
            return darkened_line_color_rgb

        # Current
        for j in range(self.robot_dynamics.n_joints):
            axs[0].plot(t, measured_output_reshaped[j, :], '-', color=plot_colors[j], linewidth=2.5,
                        label=f'joint {j + 1}, meas.')
            axs[0].plot(t, estimated_output_reshaped[j, :], color=darken_color(plot_colors[j]), linewidth=2.5,
                        label=f'joint {j + 1}, pred.')
            axs[0].legend(loc="best")
        axs[0].set_xlim([t[0], t[-1]])
        axs[0].set_title('Calibration')

        # Error
        for j in range(self.robot_dynamics.n_joints):
            axs[1].plot(t, error[j].T, '-', color=plot_colors[j], linewidth=1.3, label=f'joint {j + 1}')
            plt.legend(loc="upper left")
        axs[1].set_xlim([t[0], t[-1]])

        for ax in axs.flat:
            ax.label_outer()
        plt.setp(axs[0], ylabel='Current [A]')
        plt.setp(axs[1], ylabel='Error [A]')
        plt.setp(axs[1], xlabel='Time [s]')

        plt.show()

    def plot_calibrate_and_validate(self, parameters):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MultipleLocator
            from matplotlib.colors import hsv_to_rgb, ColorConverter, rgb_to_hsv
        except ImportError:
            import warnings
            warnings.warn("The matplotlib package is not installed, please install it for plotting the calibration.")

        ## Calibration and Validation estimation and data
        t_calibration, measured_output_reshaped, estimated_output_reshaped = self._get_plot_values_for(self.robot_data_calibration, self.gravity, parameters)
        t_validation, measured_validation_output_reshaped, validation_output_reshaped = self._get_plot_values_for(self.robot_data_validation, self.gravity, parameters)
        t_validation += t_calibration[-1]

        def darken_color(plot_color, darken_amount=0.45):
            """Computes rgb values to a darkened color"""
            line_color_rgb = ColorConverter.to_rgb(plot_color)
            line_color_hsv = rgb_to_hsv(line_color_rgb)
            darkened_line_color_hsv = line_color_hsv - np.array([0, 0, darken_amount])
            darkened_line_color_rgb = hsv_to_rgb(darkened_line_color_hsv)
            return darkened_line_color_rgb


        fig = plt.figure()
        gs = fig.add_gridspec(2, 2, hspace=0.03, wspace=0,
                                width_ratios=[np.max(t_calibration) - np.min(t_calibration),
                                            np.max(t_validation) - np.min(t_validation)])
        axs = gs.subplots(sharex='col', sharey='all')


        linewidth_meas = 2.5
        linewidth_est = 2.0
        linetype_meas = '-'
        linetype_est = '--'

        # Estimation data - current
        for j in range(self.robot_dynamics.n_joints):
            axs[0, 0].plot(t_calibration, measured_output_reshaped[j, :].T, linetype_meas, color=plot_colors[j], linewidth=linewidth_meas,
                            label=f'joint {j+1}, meas.')
            axs[0, 0].plot(t_calibration, estimated_output_reshaped[j, :].T, linetype_est, color=darken_color(plot_colors[j]), linewidth=linewidth_est, label=f'joint {j+1}, est.')
        axs[0, 0].set_xlim([t_calibration[0], t_calibration[-1]])
        axs[0, 0].set_title('Calibration')

        # Validation data - current
        mse = self.get_mse(measured_validation_output_reshaped, validation_output_reshaped)
        for j in range(self.robot_dynamics.n_joints):
            axs[0, 1].plot(t_validation, measured_validation_output_reshaped[j, :].T, linetype_meas, color=plot_colors[j], linewidth=linewidth_meas)
            axs[0, 1].plot(t_validation, validation_output_reshaped[j, :].T, linetype_est, color=darken_color(plot_colors[j]), linewidth=linewidth_est)
        axs[0, 1].set_xlim([t_validation[0], t_validation[-1]])
        axs[0, 1].set_title('Validation')

        # Estimation data - error
        error_est = (measured_output_reshaped - estimated_output_reshaped)
        for j in range(self.robot_dynamics.n_joints):
            axs[1, 0].plot(t_calibration, error_est[j].T, linetype_meas, color=plot_colors[j], linewidth=linewidth_meas, label=f'joint {j + 1}')
        axs[1, 0].set_xlim([t_calibration[0], t_calibration[-1]])

        # Validation data - error
        error_val = (measured_validation_output_reshaped - validation_output_reshaped)
        for j in range(self.robot_dynamics.n_joints):
            axs[1, 1].plot(t_validation, error_val[j].T, linetype_meas, color=plot_colors[j], linewidth=linewidth_meas, label=f'joint {j + 1}')
        axs[1, 1].set_xlim([t_validation[0], t_validation[-1]])

        # equate xtick spacing of right plots to those of left plots
        xticks_diff = axs[1, 0].get_xticks()[1] - axs[1, 0].get_xticks()[0]
        axs[1, 0].xaxis.set_major_locator(MultipleLocator(xticks_diff))
        axs[1, 1].xaxis.set_major_locator(MultipleLocator(xticks_diff))

        for ax in axs.flat:
            ax.label_outer()
        plt.setp(axs[0, 0], ylabel='Current [A]')
        plt.setp(axs[1, 0], ylabel='Error [A]')
        
        pos_label = ((len(t_validation) / len(t_calibration)) + 1) / 2
        axs[1,0].set_xlabel("Time [s]", fontsize='large', ha="center", position=(pos_label,pos_label))

        # Legend position
        axs[0, 0].legend(loc='lower left', ncol=self.robot_dynamics.n_joints)
        axs[1, 0].legend(loc="upper left", ncol=self.robot_dynamics.n_joints)
        plt.show()
    
    @staticmethod
    def _evaluate_dynamics_excitation_as_cost(observation_matrix, metric="cond"):
        assert metric in {"cond", "determinant", "log_determinant", "minimum_singular_value"}

        if metric == "cond":
            cost = np.linalg.cond(observation_matrix)
        elif metric == "determinant":
            cost = np.linalg.det(observation_matrix.T @ observation_matrix)
        elif metric == "log_determinant":
            cost = np.log(np.linalg.det(observation_matrix.T @ observation_matrix))
        elif metric == "minimum_singular_value":
            cost = np.linalg.svd(observation_matrix.T @ observation_matrix)  # validate this
        else:
            cost = 0
        return cost

    @staticmethod
    def _downsample(y, downsampling_factor):
        """The decimate procedure down-samples the signal such that the matrix system (that is later to be inverted) is not
        larger than strictly required. The signal.decimate() function can also low-pass filter the signal before
        down-sampling, but for IIR filters unfortunately only the Chebyshev filter is available which has (unwanted) ripple
        in the passband unlike the Butterworth filter that we use. The approach for downsampling is simply picking every
        downsampling_factor'th sample of the data."""

        y_ds = y[::downsampling_factor, :]
        return y_ds

    @staticmethod
    def _parallel_filter(y, dt, f_dyn):
        """Applies a 4th order Butterworth (IIR) filter for each row in y having a cutoff frequency of 2*f_dyn."""
        # Link to cut-off freq. eq.: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6151858
        parallel_filter_order = 4
        cutoff_freq_parallel = 2 * f_dyn
        parallel_filter = signal.butter(parallel_filter_order, cutoff_freq_parallel, btype='low', output='sos',
                                        fs=1 / dt)

        y_pf = signal.sosfiltfilt(parallel_filter, y, axis=0)
        return y_pf

    @staticmethod
    def _trajectory_filtering_and_central_difference(q_m, dt, f_dyn, idx_start=None, idx_end=None):
        if idx_end is not None:
            assert idx_end < q_m.shape[1], "'idx_end' must not be greater than the dataset size."

        trajectory_filter_order = 4
        cutoff_freq_trajectory = 5 * f_dyn  # Cut-off frequency should be around 5*f_dyn = 50 Hz(?)
        trajectory_filter = signal.butter(trajectory_filter_order, cutoff_freq_trajectory, btype='low', output='sos',
                                          fs=1 / dt)
        q_tf = signal.sosfiltfilt(trajectory_filter, q_m, axis=1)

        # Obtain first and second order time-derivatives of measured and filtered trajectory
        q_tf, qd_tf, qdd_tf = central_finite_difference(q_tf, dt, order=2)

        # Truncate/crop data
        # assert q_tf.shape[1] > idx_end + 1, f"'idx_end' must be smaller than or equal to the length of the signal along axis 1"
        q_tf = q_tf[:, idx_start:idx_end]
        qd_tf = qd_tf[:, idx_start:idx_end]
        qdd_tf = qdd_tf[:, idx_start:idx_end]

        assert q_tf.shape == qd_tf.shape == qdd_tf.shape, f"q_tf.shape == {q_tf.shape}, qd_tf.shape == {qd_tf.shape}, qdd_tf.shape == {qdd_tf.shape}"

        return q_tf, qd_tf, qdd_tf

    @staticmethod
    def get_mse(y_meas, y_pred):
        """
        This function calculates the mean squared error of each channel in y using sklearn.metrics.mean_squared_error.
        """
        assert y_meas.shape == y_pred.shape, f"y_meas.shape: {y_meas.shape}, y_pred.shape: {y_pred.shape}"

        n_channels = y_meas.shape[0]  # no. of channels
        mse = np.zeros(n_channels)
        for i in range(n_channels):
            mse[i] = mean_squared_error(y_meas[i, :], y_pred[i, :])
        return mse