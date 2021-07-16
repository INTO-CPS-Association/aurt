from aurt.robot_dynamics import RobotDynamics
import numpy as np
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

from aurt.file_system import cache_numpy, cache_csv, load_numpy, from_cache, cache_object, load_object
from aurt.robot_data import plot_colors


class RobotCalibration:
    def __init__(self, rd_filename, robot_data, relative_separation_of_calibration_and_prediction=None, robot_data_predict=None):

        # Load saved RobotDynamics model
        filename = from_cache(rd_filename + ".pickle")
        with open(filename, 'rb') as f:
            self.robot_dynamics: RobotDynamics = pickle.load(f)

        if relative_separation_of_calibration_and_prediction is None and robot_data_predict is None:
            self.robot_data_calibration = robot_data
            self.robot_data_prediction = None
        elif relative_separation_of_calibration_and_prediction is not None and robot_data_predict is None:
            assert 0 < relative_separation_of_calibration_and_prediction < 1, "The specified relative separation of "\
                                                                              "data used for calibration and "\
                                                                              "prediction must be in the range from 0 "\
                                                                              "to 1."

            t_sep = robot_data.time[-1] * relative_separation_of_calibration_and_prediction
            self.robot_data_calibration = robot_data
            self.robot_data_prediction = robot_data
            self.robot_data_calibration.__trim_data(desired_timeframe=(0, t_sep))
            self.robot_data_prediction.__trim_data(desired_timeframe=(t_sep, np.inf))
        elif relative_separation_of_calibration_and_prediction is None and robot_data_predict is not None:
            self.robot_data_calibration = robot_data
            self.robot_data_prediction = robot_data_predict
        else:
            print("A wrong combination of arguments was provided.")

        self.f_dyn = 10  # Approx. cut-off frequency [Hz] of robot dynamics to be estimated
        self.downsampling_factor = round(0.8 / (4 * self.f_dyn * self.robot_data_calibration.dt_nominal))
        self.parameters = None
        self.estimated_output = None
        self.number_of_samples_in_downsampled_data = None

    def __measurement_vector(self, robot_data, start_index=1, end_index=-1):
        def compute_measurement_vector():
            i = np.array([robot_data.data[f"actual_current_{j}"] for j in range(1, self.robot_dynamics.n_joints + 1)]).T
            i_pf = RobotCalibration.__parallel_filter(i, robot_data.dt_nominal, self.f_dyn)[start_index:end_index, :]
            i_pf_ds = RobotCalibration.__downsample(i_pf, self.downsampling_factor)
            return i_pf_ds.flatten(order='F')  # y = [y1, ..., yi, ..., yN],  yi = [yi_{1}, ..., yi_{n_samples}]

        return compute_measurement_vector()

    def __observation_matrix(self, robot_data, start_index=1, end_index=-1):
        def compute_observation_matrix():
            q_m = np.array([robot_data.data[f"actual_q_{j}"] for j in range(1, self.robot_dynamics.n_joints + 1)])  # (6 x n_samples) numpy array of measured angular positions

            # Low-pass filter (smoothen) measured angular position and obtain 1st and 2nd order time-derivatives
            q_tf, qd_tf, qdd_tf = RobotCalibration.__trajectory_filtering_and_central_difference(q_m,
                                                                                                 robot_data.dt_nominal,
                                                                                                 self.f_dyn,
                                                                                                 start_index,
                                                                                                 end_index)

            n_samples_ds = self.__measurement_vector(robot_data, start_index=start_index, end_index=end_index).shape[0] // self.robot_dynamics.n_joints  # No. of samples in downsampled data
            observation_matrix = np.zeros((self.robot_dynamics.n_joints * n_samples_ds, sum(self.robot_dynamics.number_of_parameters())))  # Initialization
            for j in range(self.robot_dynamics.n_joints):
                # Obtain the rows of the observation matrix related to joint j
                obs_mat_j = self.robot_dynamics.observation_matrix_joint(j, q_tf, qd_tf, qdd_tf)

                # Parallel filter and decimate/downsample the rows of the observation matrix related to joint j.
                obs_mat_j_ds = RobotCalibration.__downsample(RobotCalibration.__parallel_filter(obs_mat_j, robot_data.dt_nominal, self.f_dyn), self.downsampling_factor)
                observation_matrix[j*n_samples_ds:(j+1)*n_samples_ds, :] = obs_mat_j_ds
            return observation_matrix

        return compute_observation_matrix()

    def calibrate(self, filename_parameters):#, calibration_method='wls', weighting='variance'):
        # TODO: make it possible to specify the relative portion of the dataset you want, e.g. 0.5 for half of the
        #  dataset. Also, the actual number of e.g. 52.9 does not make any sense here - I don't think the "bias"
        #  correction of the timestamps (see the function 'load_data()' in 'data_processing.py'):
        #     time_range -= time_range[0]
        #  is corrected...
        #  EDIT: Maybe the elimination of zero-velocity data plays a role(?)

        observation_matrix = self.__observation_matrix(self.robot_data_calibration,
                                                       start_index=self.robot_data_calibration.non_static_start_index,
                                                       end_index=self.robot_data_calibration.non_static_end_index)
        measurement_vector = self.__measurement_vector(self.robot_data_calibration,
                                                       start_index=self.robot_data_calibration.non_static_start_index,
                                                       end_index=self.robot_data_calibration.non_static_end_index) 

        # sklearn fit
        OLS = LinearRegression(fit_intercept=False)
        OLS.fit(observation_matrix, measurement_vector)
        measurement_vector_ols_estimation = OLS.predict(observation_matrix)
        n_samples_est = round(len(measurement_vector) / self.robot_dynamics.n_joints)

        # Reshape measurement vector from (n_joints*n_samples x 1) to (n_samples x n_joints)
        assert n_samples_est * self.robot_dynamics.n_joints == len(measurement_vector_ols_estimation) \
               == np.shape(observation_matrix)[0]
        measurement_vector_reshape = np.reshape(measurement_vector, (self.robot_dynamics.n_joints, n_samples_est))
        measurement_vector_ols_estimation_reshape = np.reshape(measurement_vector_ols_estimation, (self.robot_dynamics.n_joints, n_samples_est))

        # Compute weights (the reciprocal of the estimated variance of the error)
        residuals = measurement_vector_reshape - measurement_vector_ols_estimation_reshape
        residual_sum_of_squares = np.sum(np.square(residuals), axis=1)
        variance_residual = residual_sum_of_squares / (n_samples_est - np.array(self.robot_dynamics.number_of_parameters()))

        # if calibration_method.lower() == 'wls':
        wls_sample_weights = np.repeat(1 / variance_residual, n_samples_est)
            # if weighting == 'variance':
            #     wls_sample_weights = np.repeat(1 / variance_residual, n_samples_est)
            # elif weighting == 'standard deviation':
            #     standard_deviation = np.sqrt(variance_residual)
            #     wls_sample_weights = np.repeat(1 / standard_deviation, n_samples_est)
        # else:
        #     wls_sample_weights = np.ones(self.robot_dynamics.n_joints, n_samples_est)

        # Weighted Least Squares solution
        wls_calibration = LinearRegression(fit_intercept=False)
        wls_calibration.fit(observation_matrix, measurement_vector, sample_weight=wls_sample_weights)

        self.parameters = wls_calibration.coef_
        
        #cache_numpy(from_cache(filename_parameters), lambda: parameters)
        cache_csv(from_cache(filename_parameters), lambda: self.parameters)
        return wls_calibration.coef_
        # return self.predict(self.robot_data_calibration, parameters, 'calibration_output')

    def predict(self, robot_data_predict, parameters, filename_predicted_output):

        def compute_prediction():
            observation_matrix = self.__observation_matrix(robot_data_predict)
            estimated_output = observation_matrix @ parameters

            n_samples_ds = round(observation_matrix.shape[0] / self.robot_dynamics.n_joints)
            assert n_samples_ds * self.robot_dynamics.n_joints == observation_matrix.shape[0]
            output_predicted_reshaped = np.reshape(estimated_output, (self.robot_dynamics.n_joints, n_samples_ds))

            return robot_data_predict.time[::self.downsampling_factor], output_predicted_reshaped  # TODO: CORRECT ERROR; 'time' does not correspond in length to 'output_predicted_reshaped'

        return cache_csv(from_cache(filename_predicted_output), compute_prediction)

    def plot_calibration(self, parameters):

        observation_matrix = self.__observation_matrix(self.robot_data_calibration)
        n_samples = observation_matrix.shape[0] // self.robot_dynamics.n_joints
        estimated_output_reshaped = np.reshape(observation_matrix @ parameters, (self.robot_dynamics.n_joints, n_samples))
        measured_output_reshaped = np.reshape(self.__measurement_vector(self.robot_data_calibration),
                                              (self.robot_dynamics.n_joints, n_samples))
        error = measured_output_reshaped - estimated_output_reshaped
        t = np.linspace(0, self.robot_data_calibration.dt_nominal*self.downsampling_factor*n_samples, n_samples)

        import matplotlib.colors
        import matplotlib.pyplot as plt
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
            axs[0].plot(t, measured_output_reshaped[j, :], '-', color=plot_colors[j], linewidth=1.5,
                           label=f'joint {j+1}, meas.')
            axs[0].plot(t, estimated_output_reshaped[j, :], color=darken_color(plot_colors[j]), linewidth=1, label=f'joint {j+1}, pred.')
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

    def plot_prediction(self, filename_predict):

        t, y = load_object(from_cache(filename_predict + '.pickle'))
        # n_samples = self.robot_data_prediction.
        # t = np.linspace(0, self.robot_data_calibration.dt_nominal * n_samples, n_samples)

        observation_matrix = self.__observation_matrix(self.robot_data_calibration)
        n_samples = observation_matrix.shape[0] // self.robot_dynamics.n_joints
        estimated_output_reshaped = np.reshape(observation_matrix @ parameters,
                                               (self.robot_dynamics.n_joints, n_samples))
        measured_output_reshaped = np.reshape(self.__measurement_vector(self.robot_data_calibration),
                                              (self.robot_dynamics.n_joints, n_samples))
        error = measured_output_reshaped - estimated_output_reshaped
        t = np.linspace(0, self.robot_data_calibration.dt_nominal * self.downsampling_factor * n_samples, n_samples)

        import matplotlib.colors
        import matplotlib.pyplot as plt
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
            axs[0].plot(t, measured_output_reshaped[j, :], '-', color=plot_colors[j], linewidth=1.5,
                        label=f'joint {j}, meas.')
            axs[0].plot(t, estimated_output_reshaped[j, :], color=darken_color(plot_colors[j]), linewidth=1,
                        label=f'joint {j}, pred.')
        axs[0].set_xlim([t[0], t[-1]])
        axs[0].set_title('Prediction')

        # Error
        for j in range(self.robot_dynamics.n_joints):
            axs[1].plot(t, error[j].T, '-', color=plot_colors[j], linewidth=1.3, label=f'joint {j + 1}')
        axs[1].set_xlim([t[0], t[-1]])

        for ax in axs.flat:
            ax.label_outer()
        plt.setp(axs[0], ylabel='Current [A]')
        plt.setp(axs[1], ylabel='Error [A]')
        plt.setp(axs[1], xlabel='Time [s]')

        plt.show()

    def plot_estimation_and_prediction(self, filename_predict):
        t_prediction, output_predicted_reshaped = load_numpy(filename_predict + '.npy')
        t_calibration = self.robot_data_calibration.time

        import matplotlib.pyplot as plt
        fig = plt.figure()
        gs = fig.add_gridspec(2, 2, hspace=0.03, wspace=0,
                              width_ratios=[np.max(t_calibration) - np.min(t_calibration), np.max(t_prediction) - np.min(t_prediction)])
        axs = gs.subplots(sharex='col', sharey='all')

        fig.supxlabel('Time [s]')
        fig.supylabel('Current [A]')

        # Estimation data - current
        for j in range(self.robot_dynamics.n_joints):
            axs[0, 0].plot(self.robot_data.time, y_est_reshape[j, :].T, '-', color=plot_colors[j], linewidth=1.3,
                           label=f'joint {j}, meas.')
            axs[0, 0].plot(t_est, y_wls_est_reshape[j, :].T, color='k', linewidth=0.6, label=f'joint {j}, pred.')
        axs[0, 0].set_xlim([t_est[0], t_est[-1]])
        axs[0, 0].set_title('Estimation')

        # Validation data - current
        for j in range(self.robot_dynamics.n_joints):
            axs[0, 1].plot(t_val, y_val_reshape[j, :].T, '-', color=plot_colors[j], linewidth=1.3,
                           label=f'joint {j}, meas.')
            axs[0, 1].plot(t_val, y_wls_val_reshape[j, :].T, color='k', linewidth=0.6,
                           label=f'joint {j}, pred. (mse: {mse[j]:.3f})')
        axs[0, 1].set_xlim([t_val[0], t_val[-1]])
        axs[0, 1].set_title('Validation')

        # Estimation data - error
        error_est = (y_est_reshape - y_wls_est_reshape)
        for j in range(self.robot_dynamics.n_joints):
            axs[1, 0].plot(t_est, error_est[j].T, '-', color=plot_colors[j], linewidth=1.3, label=f'joint {j + 1}')
        axs[1, 0].set_xlim([t_est[0], t_est[-1]])

        # Validation data - error
        error_val = (y_val_reshape - y_wls_val_reshape)
        for j in range(self.robot_dynamics.n_joints):
            axs[1, 1].plot(t_val, error_val[j].T, '-', color=plot_colors[j], linewidth=1.3, label=f'joint {j + 1}')
        axs[1, 1].set_xlim([t_val[0], t_val[-1]])

        # equate xtick spacing of right plots to those of left plots
        xticks_diff = axs[1, 0].get_xticks()[1] - axs[1, 0].get_xticks()[0]
        axs[1, 0].xaxis.set_major_locator(MultipleLocator(xticks_diff))
        axs[1, 1].xaxis.set_major_locator(MultipleLocator(xticks_diff))

        for ax in axs.flat:
            ax.label_outer()
        plt.setp(axs[0, 0], ylabel='Signal')
        plt.setp(axs[1, 0], ylabel='Error')

        # Legend position
        l_val = (np.max(t_val) - np.min(t_val))
        l_tot = (np.max(t_val) - np.min(t_est))
        l_val_rel = l_val / l_tot
        legend_x_position = 1 - 0.5 / l_val_rel  # global center of legend as seen relative to the validation dataset
        if not NONINTERACTIVE:
            axs[0, 1].legend(loc='lower center', bbox_to_anchor=(legend_x_position, -0.022), ncol=Njoints)
            plt.show()
        # **************************************************************************************************************
        return 1

    @staticmethod
    def __downsample(y, downsampling_factor):
        """The decimate procedure down-samples the signal such that the matrix system (that is later to be inverted) is not
        larger than strictly required. The signal.decimate() function can also low-pass filter the signal before
        down-sampling, but for IIR filters unfortunately only the Chebyshev filter is available which has (unwanted) ripple
        in the passband unlike the Butterworth filter that we use. The approach for downsampling is simply picking every
        downsampling_factor'th sample of the data."""

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
    def __trajectory_filtering_and_central_difference(q_m, dt, f_dyn, idx_start=1, idx_end=-1):
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
        print(f"idx_start: {idx_start}")
        print(f"idx_end: {idx_end}")
        print(f"q_tf.shape: {q_tf.shape}")
        print(f"qd_tf.shape: {qd_tf.shape}")
        print(f"qdd_tf.shape: {qdd_tf.shape}")
        q_tf = q_tf[:, idx_start:idx_end]
        qd_tf = qd_tf[:, idx_start:idx_end]
        qdd_tf = qdd_tf[:, idx_start - 1:idx_end - 1]  # shifted due to a "lost" index in the start of the dataset
        print(f"q_tf.shape: {q_tf.shape}")
        print(f"qd_tf.shape: {qd_tf.shape}")
        print(f"qdd_tf.shape: {qdd_tf.shape}")

        assert q_tf.shape == qd_tf.shape == qdd_tf.shape, f"q_tf.shape == {q_tf.shape}, qd_tf.shape == {qd_tf.shape}, qdd_tf.shape == {qdd_tf.shape}"

        return q_tf, qd_tf, qdd_tf

    @staticmethod
    def get_mse(y_meas, y_pred):
        """
        This function calculates the mean squared error of each channel in y using sklearn.metrics.mean_squared_error.
        """
        assert y_meas.shape == y_pred.shape

        n_channels = y_meas.shape[0]  # no. of channels
        mse = np.zeros(n_channels)
        for i in range(n_channels):
            mse[i] = mean_squared_error(y_meas[i, :], y_pred[i, :])
        return mse
