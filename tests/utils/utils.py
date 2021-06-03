from genericpath import isfile
from math import pi
import pandas as pd
from sys import setrecursionlimit
from os.path import isfile
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression

from aurt.num_sym_layers import *
from aurt.file_system import cache_object, load_numpy_expr, store_numpy_expr, from_project_root
from aurt.data_processing import plot_colors

from tests.linearization_tests import compute_regressor_with_instantiated_parameters, compute_indices_base_exist, compute_observation_matrix_and_measurement_vector, compute_parameters_base, get_mse
from tests import NONINTERACTIVE


# TODO: clean code after this point
Njoints = 6
q = [0.0] + [sp.symbols(f"q{j}") for j in range(1, Njoints + 1)]
qd = [0.0] + [sp.symbols(f"qd{j}") for j in range(1, Njoints + 1)]
qdd = [0.0] + [sp.symbols(f"qdd{j}") for j in range(1, Njoints + 1)]


# TODO: Needs to be tested.
def calibration(mdh_params_func):
    t_est_val_separation = 52.5  # timely separation of estimation and validation datasets, TODO: make it possible to specify the relative portion of the dataset you want, e.g. 0.5 for half of the dataset.

    data_id = f"random_motion_{t_est_val_separation}"
    observation_matrix_file_estimation = from_project_root(f'tests/cache/observation_matrix_estimation_{data_id}.npy')
    measurement_vector_file_estimation = from_project_root(f'tests/cache/measurement_vector_estimation_{data_id}.npy')
    observation_matrix_file_validation = from_project_root(f'tests/cache/observation_matrix_validation_{data_id}.npy')
    measurement_vector_file_validation = from_project_root(f'tests/cache/measurement_vector_validation_{data_id}.npy')

    setrecursionlimit(int(1e6))  # Prevents errors in sympy lambdify
    args_sym = q[1:] + qd[1:] + qdd[1:]  # list concatenation
    regressor_reduced_func = sp.lambdify(args_sym,
                                            compute_regressor_with_instantiated_parameters(
                                                ur_param_function=mdh_params_func), 'numpy')
    filename = from_project_root("tests/cache/parameter_indices_base")
    idx_base_global = cache_object(filename, lambda: compute_indices_base_exist(regressor_reduced_func))
    filename = from_project_root('tests/cache/regressor_base_with_instantiated_parameters')
    regressor_base_params = cache_object(filename, lambda: compute_regressor_with_instantiated_parameters(
        ur_param_function=mdh_params_func)[1:, idx_base_global])

    if not isfile(observation_matrix_file_estimation):
        # The base parameter system is obtained by passing only the 'idx_base' columns of the regressor
        W_est, y_est = compute_observation_matrix_and_measurement_vector(
            from_project_root(f'resources/Dataset/ur5e_all_joints_same_time/random_motion.csv'),
            regressor_base_params,
            time_frame=(-np.inf, t_est_val_separation))
        W_val, y_val = compute_observation_matrix_and_measurement_vector(
            from_project_root(f'resources/Dataset/ur5e_all_joints_same_time/random_motion.csv'),
            regressor_base_params,
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

    # Check output, i.e. evaluate the ew. i_measured - observation_matrix * p_num
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
    # covariance_matrix_residual = np.diag(np.repeat(variance_residual, n_samples_est))
    # inverse_covariance_matrix_residual = np.diag(np.repeat(1/variance_residual, n_samples_est))

    WLS_sample_weights = np.repeat(1/variance_residual, n_samples_est)  # Currently, no weights are used -> OLS estimation

    # Weighted Least Squares solution
    WLS = LinearRegression(fit_intercept=False)
    WLS.fit(W_est, y_est, sample_weight=WLS_sample_weights)
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

    #fig.supxlabel('Time [s]')
    #fig.supylabel('Current [A]')

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
