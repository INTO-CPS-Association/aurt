import os
from itertools import product, chain

import sys
import unittest
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import sympy as sp
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from aurt.calibration_aux import find_nonstatic_start_and_end_indices
from aurt.robot_data import RobotData, plot_colors
from aurt.file_system import cache_object, store_object, load_object, project_root
from aurt.num_sym_layers import spzeros_array, spvector, npzeros_array
from aurt.robot_dynamics import RobotDynamics
from aurt.robot_calibration import RobotCalibration
from aurt.data_processing import convert_file_to_mdh
from aurt.joint_dynamics import JointDynamics
from tests import NONINTERACTIVE
from tests.utils.timed_test import TimedTest

"""
# ************************************************** NOT IN USE ATM ****************************************************
def gms_friction(j):
    # NOT VALIDATED
    assert 0 < n_gms_elements < max_gms_elements

    # Deltaj = [np.power(float(i)/n_elements, gms_spacing_power_law) for i in range(n_elements)]
    #
    # for i in range(n_elements):
    #     z1[j][i] = sp.sign(z0j[i] + dq[j]) * min(abs(z0j[i] + dq[j]), backlash[j]*Deltaj[i])  # State equation
    #
    # yj = k_gms[j].T * z1[j] / backlash[j]  # Friction torque
    #
    # return yj


def compute_gms_states(dqj, z0j, backlashj):
    # NOT VALIDATED
    n_samples = dqj.size
    z1j = np.zeros_like(z0j)
    n_elements = z1j.shape[1]
    spacing_power_law = 4.0
    Deltaj = [np.power(float(i) / n_elements, spacing_power_law) for i in range(n_elements)]

    for i in range(n_elements):
        z1j[i] = np.sign(z0j[i] + dqj) * np.min(np.abs(z0j[i] + dqj), backlashj*Deltaj[i])  # State equation

    return z1j
# **********************************************************************************************************************
"""


def check_symbolic_linear_system(tau, regressor_matrix, parameter_vector, joints=None):
    """Symbolically checks that the regressor matrix times the parameter vector equals tau"""
    if joints is None:
        joints = list(range(1, Njoints+1))

    for j in joints:
        reg_lin_mul_j = regressor_matrix[j, :].dot(parameter_vector)
        assert sp.simplify(tau[j] - reg_lin_mul_j) == 0, f"Joint {j}: FAIL!"


class LinearizationTests(TimedTest):
    def test_calibration_new(self):
        mdh_filepath = "C:/sourcecontrol/github/aurt/resources/robot_parameters/ur3e_params.csv"
        mdh = convert_file_to_mdh(mdh_filepath)
        my_robot_dynamics = RobotDynamics(mdh)

        robot_data_path = os.path.join(project_root(), 'resources', 'Dataset', 'ur5e_all_joints_same_time', 'random_motion.csv')
        t_est_val_separation = 63.0
        filename_parameters = 'parameters'
        my_robot_calibration_data = RobotData(robot_data_path,
                                              delimiter=' ',
                                              desired_timeframe=(-np.inf, t_est_val_separation),
                                              interpolate_missing_samples=True)
        my_robot_calibration = RobotCalibration(my_robot_dynamics, my_robot_calibration_data)
        parameters = my_robot_calibration.calibrate(filename_parameters)
        my_robot_calibration.plot_calibration(parameters)

        filename_predicted_output = 'predicted_output'
        # my_robot_validation_data = RobotData(robot_data_path,
        #                                      delimiter=' ',
        #                                      desired_timeframe=(t_est_val_separation, np.inf),
        #                                      interpolate_missing_samples=True)
        # t, y_pred = my_robot_calibration.predict(my_robot_validation_data, filename_parameters, filename_predicted_output)
        # my_robot_calibration.plot_prediction(filename_predicted_output)

    def test_calibration_ur5e_45deg_base(self):
        mdh_filepath = "C:/sourcecontrol/github/aurt/resources/robot_parameters/ur5e_params.csv"
        mdh = convert_file_to_mdh(mdh_filepath)
        my_robot_dynamics = RobotDynamics(mdh)

        robot_data_path = os.path.join(project_root(), 'resources', 'Dataset', 'ur5e_all_joints_same_time', 'calibration_motion.csv')
        filename_parameters = 'parameters'
        my_robot_calibration_data = RobotData(robot_data_path,
                                              delimiter=' ',
                                              interpolate_missing_samples=True)
        my_robot_calibration = RobotCalibration(my_robot_dynamics, my_robot_calibration_data)
        parameters = my_robot_calibration.calibrate(filename_parameters)
        my_robot_calibration.plot_calibration(parameters)

        filename_predicted_output = 'predicted_output'
        # my_robot_validation_data = RobotData(robot_data_path,
        #                                      delimiter=' ',
        #                                      desired_timeframe=(t_est_val_separation, np.inf),
        #                                      interpolate_missing_samples=True)
        # t, y_pred = my_robot_calibration.predict(my_robot_validation_data, filename_parameters, filename_predicted_output)
        # my_robot_calibration.plot_prediction(filename_predicted_output)

    def test_calibrate_parameters(self):
        t_est_val_separation = 63.0  # timely separation of estimation and validation datasets
        # TODO: make it possible to specify the relative portion of the dataset you want, e.g. 0.5 for half of the
        #  dataset. Also, the actual number of e.g. 52.9 does not make any sense here - I don't think the "bias"
        #  correction of the timestamps (see the function 'load_data()' in 'data_processing.py'):
        #     time_range -= time_range[0]
        #  is corrected...
        #  EDIT: Maybe the elimination of zero-velocity data plays a role(?)

        data_id = f"random_motion_{t_est_val_separation}"
        observation_matrix_file_estimation = f'./observation_matrix_estimation_{data_id}.npy'
        measurement_vector_file_estimation = f'./measurement_vector_estimation_{data_id}.npy'
        observation_matrix_file_validation = f'./observation_matrix_validation_{data_id}.npy'
        measurement_vector_file_validation = f'./measurement_vector_validation_{data_id}.npy'

        # sys.setrecursionlimit(int(1e6))  # Prevents errors in sympy lambdify
        # args_sym = q[1:] + qd[1:] + qdd[1:]  # list concatenation
        # if load_model.lower() != 'none':
            # args_sym += tauJ[1:]

        # regressor_reduced_func = sp.lambdify(args_sym, compute_regressor_with_instantiated_parameters(
        #     ur_param_function=get_ur5e_parameters), 'numpy')
        # filename = "parameter_indices_base"
        # idx_base_global = cache_object(filename, lambda: compute_indices_base_exist(regressor_reduced_func))
        # filename = 'regressor_base_with_instantiated_parameters'
        # regressor_base_params = cache_object(filename, lambda: compute_regressor_with_instantiated_parameters(
        #     ur_param_function=get_ur5e_parameters)[1:, idx_base_global])

        regressor_base_params_2 = RobotDynamics(None).regressor()

        # TODO: manually using 'store_numpy_expr' and 'load_numpy_expr' - why not use 'cache_numpy' instead?
        if not os.path.isfile(observation_matrix_file_estimation):
            # The base parameter system is obtained by passing only the 'idx_base' columns of the regressor
            root_dir = project_root()
            W_est, y_est = compute_observation_matrix_and_measurement_vector(
                os.path.join(root_dir, 'resources', 'Dataset', 'ur5e_all_joints_same_time', 'random_motion.csv'),  #'aurt/resources/Dataset/ur5e_all_joints_same_time/random_motion.csv',
                regressor_base_params_2,
                time_frame=(-np.inf, t_est_val_separation))
            W_val, y_val = compute_observation_matrix_and_measurement_vector(
                os.path.join(root_dir, 'resources', 'Dataset', 'ur5e_all_joints_same_time', 'random_motion.csv'),
                regressor_base_params_2,
                time_frame=(t_est_val_separation, np.inf))
            store_numpy_expr(W_est, observation_matrix_file_estimation)
            store_numpy_expr(y_est, measurement_vector_file_estimation)
            store_numpy_expr(W_val, observation_matrix_file_validation)
            store_numpy_expr(y_val, measurement_vector_file_validation)
        else:
            W_est = load_numpy_expr(observation_matrix_file_estimation)
            y_est = load_numpy_expr(measurement_vector_file_estimation)
            W_val = load_numpy_expr(observation_matrix_file_validation)
            y_val = load_numpy_expr(measurement_vector_file_validation)

        # ********************************************** NOT USED ATM **************************************************
        # cond = evaluate_observation_matrix_cost(W, metric="cond")
        # print(f"The condition number of the observation matrix is {cond}")
        # **************************************************************************************************************

        # sklearn fit
        OLS = LinearRegression(fit_intercept=False)
        OLS.fit(W_est, y_est)

        # Check output, i.e. evaluate i_measured - observation_matrix * p_num
        y_ols_est = OLS.predict(W_est)
        n_samples_est = round(len(y_est) / Njoints)
        n_samples_val = round(len(y_val) / Njoints)

        # Reshape measurement vector from (Njoints*n_samples x 1) to (n_samples x Njoints)
        assert n_samples_est*Njoints == len(y_ols_est) and n_samples_est*Njoints == np.shape(W_est)[0]
        y_est_reshape = np.reshape(y_est, (Njoints, n_samples_est))
        y_val_reshape = np.reshape(y_val, (Njoints, n_samples_val))
        y_est_ols_reshape = np.reshape(y_ols_est, (Njoints, n_samples_est))

        # Compute weights (the reciprocal of the estimated standard deviation of the error)
        idx_base, n_par_base, p_base = compute_parameters_base()
        residuals = y_est_reshape - y_est_ols_reshape
        residual_sum_of_squares = np.sum(np.square(residuals), axis=1)
        variance_residual = residual_sum_of_squares / (n_samples_est - np.array(n_par_base[1:]))
        # standard_deviation_residual = np.sqrt(variance_residual)

        wls_sample_weights = np.repeat(1/variance_residual, n_samples_est)

        # Weighted Least Squares solution
        WLS = LinearRegression(fit_intercept=False)
        WLS.fit(W_est, y_est, sample_weight=wls_sample_weights)
        y_wls_est = WLS.predict(W_est)
        y_wls_val = WLS.predict(W_val)
        y_wls_est_reshape = np.reshape(y_wls_est, (Njoints, n_samples_est))
        y_wls_val_reshape = np.reshape(y_wls_val, (Njoints, n_samples_val))

        # ************************************************** PLOTTING **************************************************
        mse = get_mse(y_val_reshape, y_wls_val_reshape)
        print(f"MSE: {mse}")

        t_est = np.linspace(0, n_samples_est-1, n_samples_est)*0.002  # TODO: remove hardcoded dt=0.002
        t_val = np.linspace(n_samples_est, n_samples_est + n_samples_val - 1, n_samples_val) * 0.002  # TODO: remove hardcoded dt=0.002
        fig = plt.figure()
        gs = fig.add_gridspec(2, 2, hspace=0.03, wspace=0, width_ratios=[np.max(t_est)-np.min(t_est), np.max(t_val)-np.min(t_val)])
        axs = gs.subplots(sharex='col', sharey='all')

        fig.supxlabel('Time [s]')
        fig.supylabel('Current [A]')

        # Estimation data - current
        for j in range(Njoints):
            axs[0, 0].plot(t_est, y_est_reshape[j, :].T, '-', color=plot_colors[j], linewidth=1.3, label=f'joint {j}, meas.')
            axs[0, 0].plot(t_est, y_wls_est_reshape[j, :].T, color='k', linewidth=0.6, label=f'joint {j}, pred.')
        axs[0, 0].set_xlim([t_est[0], t_est[-1]])
        axs[0, 0].set_title('Estimation')

        # Validation data - current
        for j in range(Njoints):
            axs[0, 1].plot(t_val, y_val_reshape[j, :].T, '-', color=plot_colors[j], linewidth=1.3, label=f'joint {j}, meas.')
            axs[0, 1].plot(t_val, y_wls_val_reshape[j, :].T, color='k', linewidth=0.6, label=f'joint {j}, pred. (mse: {mse[j]:.3f})')
        axs[0, 1].set_xlim([t_val[0], t_val[-1]])
        axs[0, 1].set_title('Validation')

        # Estimation data - error
        error_est = (y_est_reshape - y_wls_est_reshape)
        for j in range(Njoints):
            axs[1, 0].plot(t_est, error_est[j].T, '-', color=plot_colors[j], linewidth=1.3, label=f'joint {j+1}')
        axs[1, 0].set_xlim([t_est[0], t_est[-1]])

        # Validation data - error
        error_val = (y_val_reshape - y_wls_val_reshape)
        for j in range(Njoints):
            axs[1, 1].plot(t_val, error_val[j].T, '-', color=plot_colors[j], linewidth=1.3, label=f'joint {j+1}')
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
        legend_x_position = 1 - 0.5/l_val_rel  # global center of legend as seen relative to the validation dataset
        if not NONINTERACTIVE:
            axs[0, 1].legend(loc='lower center', bbox_to_anchor=(legend_x_position, -0.022), ncol=Njoints)
            plt.show()
        # **************************************************************************************************************

    def test_linear_torques(self):
        tau_sym_linearizable = cache_object('./tau_sym_linearizable', compute_tau_sym_linearizable)

        # Checking that tau_sym_linearizable is linearizable wrt. the parameter vector
        diff_p = sp.zeros(Njoints + 1, sum(n_par_linear))
        for j in range(1, Njoints + 1):  # joint loop
            for i in range(n_par_linear[j]):  # range(n_par_j_rbd_revolute + n_par_j_joint_dynamics):
                column_idx = (j - 1) * (n_par_j_rbd_revolute + n_par_j_joint_dynamics) + i
                diff_p[j, column_idx] = sp.diff(tau_sym_linearizable[j], p_linear[j][i])
                if not NONINTERACTIVE:
                    print(f"Checking that the dynamics for joint {j} is linear with respect to the parameter {p_linear[j][i]}...")
                assert p_linear[j][i] not in diff_p[j, column_idx].free_symbols

    def test_check_reduced_regressor_linear(self):
        tau_sym_linearizable = cache_object('./tau_sym_linearizable', compute_tau_sym_linearizable)
        regressor = sp.sympify(cache_object('./regressor', compute_regressor_parallel))
        regressor_reduced, _ = cache_object('./regressor_reduced', compute_regressor_linear_exist)

        parameter_vector = sp.Matrix(sp.Matrix(p_linear)[:])

        idx_exist, _, _ = compute_parameters_linear_exist(regressor)

        check_symbolic_linear_system(tau_sym_linearizable, regressor, parameter_vector)
        check_symbolic_linear_system(tau_sym_linearizable, regressor_reduced, parameter_vector[idx_exist])


if __name__ == '__main__':
    unittest.main()
