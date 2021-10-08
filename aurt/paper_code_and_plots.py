
import pickle

from aurt import api
from aurt.robot_calibration import RobotCalibration
from aurt.robot_data import RobotData
from aurt.data_processing import *
from aurt.file_system import from_cache
import numpy as np

def plot_calibration_paper(rc, parameters, validation_data):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
        from matplotlib.colors import hsv_to_rgb, ColorConverter, rgb_to_hsv
    except ImportError:
        import warnings
        warnings.warn("The matplotlib package is not installed, please install it for plotting the calibration.")

    ## Calibration and Validation estimation and data
    t_calibration, measured_output_reshaped, estimated_output_reshaped = rc._get_plot_values_for(rc.robot_data_calibration, parameters)
    t_validation, measured_validation_output_reshaped, validation_output_reshaped = rc._get_plot_values_for(validation_data, parameters)

    def darken_color(plot_color, darken_amount=0.35):
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

    #fig.supxlabel('Time [s]')

    linewidth_meas = 1.3
    linewidth_est = 1
    linetype_meas = '-'
    linetype_est = '--'

    # Estimation data - current
    for j in range(rc.robot_dynamics.n_joints):
        axs[0, 0].plot(t_calibration, measured_output_reshaped[j, :].T, linetype_meas, color=plot_colors[j], linewidth=linewidth_meas,
                        label=f'joint {j+1}, meas.')
        axs[0, 0].plot(t_calibration, estimated_output_reshaped[j, :].T, linetype_est, color=darken_color(plot_colors[j]), linewidth=linewidth_est, label=f'joint {j+1}, est.')
    axs[0, 0].set_xlim([t_calibration[0], t_calibration[-1]])
    axs[0, 0].set_title('Calibration')

    # Validation data - current
    mse = rc.get_mse(measured_validation_output_reshaped, validation_output_reshaped)
    for j in range(rc.robot_dynamics.n_joints):
        # axs[0, 1].plot(t_validation, measured_validation_output_reshaped[j, :].T, linetype_meas, color=plot_colors[j], linewidth=linewidth_meas,
        #                label=f'joint {j+1}, meas.')
        # axs[0, 1].plot(t_validation, validation_output_reshaped[j, :].T, linetype_est, color=darken_color(plot_colors[j]), linewidth=linewidth_est,
        #                label=f'joint {j+1}, est. (mse: {mse[j]:.3f})')
        axs[0, 1].plot(t_validation, measured_validation_output_reshaped[j, :].T, linetype_meas, color=plot_colors[j], linewidth=linewidth_meas)
        axs[0, 1].plot(t_validation, validation_output_reshaped[j, :].T, linetype_est, color=darken_color(plot_colors[j]), linewidth=linewidth_est)
    axs[0, 1].set_xlim([t_validation[0], t_validation[-1]])
    axs[0, 1].set_title('Validation')

    # Estimation data - error
    error_est = (measured_output_reshaped - estimated_output_reshaped)
    for j in range(rc.robot_dynamics.n_joints):
        axs[1, 0].plot(t_calibration, error_est[j].T, linetype_meas, color=plot_colors[j], linewidth=linewidth_meas, label=f'joint {j + 1}')
    axs[1, 0].set_xlim([t_calibration[0], t_calibration[-1]])

    # Validation data - error
    error_val = (measured_validation_output_reshaped - validation_output_reshaped)
    for j in range(rc.robot_dynamics.n_joints):
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
    print(pos_label)
    axs[1,0].set_xlabel("Time [s]", fontsize='large', ha="center", position=(pos_label,pos_label))

    # from matplotlib import rcParams
    # rcParams['xtick.major.pad']='20'
    # rcParams['ytick.major.pad']='8'

    # Legend position
    axs[0, 0].legend(loc='lower left', ncol=rc.robot_dynamics.n_joints)
    axs[1,0].legend(loc="upper left", ncol=rc.robot_dynamics.n_joints)
    plt.show()

def plot_prediction_paper(rc,parameters, validation_data):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
        from matplotlib.colors import hsv_to_rgb, ColorConverter, rgb_to_hsv
    except ImportError:
        import warnings
        warnings.warn("The matplotlib package is not installed, please install it for plotting the calibration.")

    ## Calibration and Validation estimation and data
    t_validation, measured_validation_output_reshaped, validation_output_reshaped = rc._get_plot_values_for(validation_data, parameters)
    error = measured_validation_output_reshaped - validation_output_reshaped

    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, hspace=0.03)
    axs = gs.subplots(sharex='col', sharey='all')

    def darken_color(plot_color, darken_amount=0.35):
        """Computes rgb values to a darkened color"""
        line_color_rgb = ColorConverter.to_rgb(plot_color)
        line_color_hsv = rgb_to_hsv(line_color_rgb)
        darkened_line_color_hsv = line_color_hsv - np.array([0, 0, darken_amount])
        darkened_line_color_rgb = hsv_to_rgb(darkened_line_color_hsv)
        return darkened_line_color_rgb

    # Current
    #for j in range(rc.robot_dynamics.n_joints):
    joint = 1
    axs[0].plot(t_validation, measured_validation_output_reshaped[joint, :], '-', color=plot_colors[joint], linewidth=1.5,
                    label=f'joint {joint+1}, meas.')
    axs[0].plot(t_validation, validation_output_reshaped[joint, :], color=darken_color(plot_colors[joint]), linewidth=1,
                    label=f'joint {joint+1}, pred.')
    axs[0].set_xlim([t_validation[0], t_validation[-1]])
    axs[0].set_title('Prediction')

    # Error
    #for j in range(rc.robot_dynamics.n_joints):
    axs[1].plot(t_validation, error[joint].T, '-', color=plot_colors[joint], linewidth=1.3, label=f'joint {joint + 1}')
    axs[1].set_xlim([t_validation[0], t_validation[-1]])

    for ax in axs.flat:
        ax.label_outer()
    plt.setp(axs[0], ylabel='Current [A]')
    plt.setp(axs[1], ylabel='Error [A]')
    plt.setp(axs[1], xlabel='Time [s]')

    plt.show()

def plot_friction_models(joint, time, measured_data, square_data, absolute_data, none_data):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
        from matplotlib.colors import hsv_to_rgb, ColorConverter, rgb_to_hsv
    except ImportError:
        import warnings
        warnings.warn("The matplotlib package is not installed, please install it for plotting the calibration.")


    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, hspace=0.03)
    axs = gs.subplots(sharex='col')

    meas_joint = 1
    square_color = meas_joint+7
    abs_color = meas_joint+6
    none_color = meas_joint+5

    # Current
    print(f"measured_data.shape {measured_data.shape}")
    axs[0].plot(time, measured_data[joint, :], '-', color=plot_colors[meas_joint], linewidth=2.0,
                    label=f'joint {joint+1}')
    axs[0].plot(time, square_data[joint, :], '-', color=plot_colors[square_color], linewidth=1.5,
                    label=f'square')
    axs[0].plot(time, absolute_data[joint, :], '-', color=plot_colors[abs_color], linewidth=1.5,
                    label=f'absolute')
    axs[0].plot(time, none_data[joint, :], '-', color=plot_colors[none_color], linewidth=1.5,
                    label=f'none')
    axs[0].set_xlim([time[0], time[-1]])
    # axs[0].set_title('Joint Dynamics Models')

    # Error
    square_error = square_data - measured_data
    absolute_error = absolute_data - measured_data
    none_error = none_data - measured_data
    axs[1].plot(time, square_error[joint].T, '-', color=plot_colors[square_color], linewidth=1.3, label=f'square')
    axs[1].plot(time, absolute_error[joint].T, '-', color=plot_colors[abs_color], linewidth=1.3, label=f'absolute')
    axs[1].plot(time, none_error[joint].T, '-', color=plot_colors[none_color], linewidth=1.3, label=f'none')
    axs[1].set_xlim([time[0], time[-1]])
    axs[1].set_ylim([-0.42, 0.42])
    

    for ax in axs.flat:
        ax.label_outer()
    plt.setp(axs[0], ylabel='Current [A]')
    plt.setp(axs[1], ylabel='Error [A]')
    plt.setp(axs[1], xlabel='Time [s]')

    axs[0].legend(loc='upper center', ncol=4)
    # axs[1].legend(loc="upper center", ncol=3)

    plt.show()



def calibrate_paper(rd_model_path, output_calibration, data_path_calibration):
    #data_path_calibration = "resources/Dataset/paper_plot/friction_test_j1.csv"
    #data_path_calibration = "resources/Dataset/onelink_data.csv"
    output_params = "calibration_params_paper.csv"
    rc = RobotCalibration(rd_model_path, data_path_calibration)
    params = rc.calibrate(output_params)

    pathfile = from_cache(output_calibration + ".pickle")
    with open(pathfile, 'wb') as f:
        pickle.dump(rc, f)

    return params, rc

def calibrate_and_validation_plot(rd_model_path, output_calibration, validation_filename, data_path_calibration):
    params, rc = calibrate_paper(rd_model_path, output_calibration, data_path_calibration)
    rc_predict_data = RobotData(validation_filename, delimiter=' ', interpolate_missing_samples=True)
    plot_calibration_paper(rc, params,rc_predict_data)


def rd_and_calibrate_for_friction_model(friction_torque_model, rd_filename, output_calibration, data_path_calibration):
    #rbd_filename = "ur5e_45deg_rbd"
    rbd_filename = "onelink_rbd"
    friction_viscous_powers = [1] #[1, 2, 3]
    api.compile_rd(rbd_filename, friction_torque_model, friction_viscous_powers, rd_filename)
    params, rc = calibrate_paper(rd_filename, output_calibration, data_path_calibration)
    return params, rc


def friction_prediction_plot(test):
    if test == "base":
        print("base")
        filename = "resources/Dataset/paper_plot/friction_test_j1.csv"
        joint = 0
    elif test == "shoulder":
        print("Shoulder")
        filename = "resources/Dataset/paper_plot/friction_test_j2.csv"
        joint = 0
    else:
        print("elbow")
        filename = "resources/Dataset/paper_plot/friction_test_j3.csv"
        joint = 0
    rc_predict_data = RobotData(filename, delimiter=' ', interpolate_missing_samples=True)

    # Calculate for each friction load model
    friction_torque_models = ["square","abs","none"]
    rd_filenames = ["rd_friction_square", "rd_friction_absolute", "rd_friction_none"]
    calibration_filenames = ["calibration_square","calibration_absolute","calibration_none"]


    square_params, square_rc = rd_and_calibrate_for_friction_model(friction_torque_models[0], rd_filenames[0], calibration_filenames[0], filename)
    absolute_params, absolute_rc = rd_and_calibrate_for_friction_model(friction_torque_models[1], rd_filenames[1], calibration_filenames[1], filename)
    none_params, none_rc = rd_and_calibrate_for_friction_model(friction_torque_models[2], rd_filenames[2], calibration_filenames[2], filename)
    
    t_square_data, measured_square_data, estimated_square_data = square_rc._get_plot_values_for(rc_predict_data, square_params)
    t_absolute_data, measured_absolute_data, estimated_absolute_data = absolute_rc._get_plot_values_for(rc_predict_data, absolute_params)
    t_none_data, measured_none_data, estimated_none_data = none_rc._get_plot_values_for(rc_predict_data, none_params)

    print(f"shape of measured data: {measured_absolute_data.shape}")

    plot_friction_models(joint, t_square_data, measured_square_data, estimated_square_data, estimated_absolute_data, estimated_none_data)
    


if __name__=='__main__':
    choice = int(input("Choose what you want to run and plot:\n"))
    run_choices = {1: "calibrate and validation", 
                    2: "base friction prediction",
                    3: "shoulder friction load models prediction",
                    4: "elbow friction load models prediction"}


    recompile_rbd = True

    if recompile_rbd == True:
        print("Recompiling rbd")
        api.compile_rbd("resources/robot_parameters/onelink_dh.csv", [0.0, -9.81, 0], "onelink_rbd", False)

    if choice == 2:
        print(f"Choice 2: {run_choices[choice]}")
        friction_prediction_plot("base")
    elif choice == 3:
        print(f"Choice 3: {run_choices[choice]}")
        friction_prediction_plot("shoulder")
    elif choice == 4:
        print(f"Choice 4: {run_choices[choice]}")
        friction_prediction_plot("elbow")

    # if recompile_rbd == True:
    #     print("Recompiling rbd")
    #     api.compile_rbd("resources/robot_parameters/ur5e_dh.csv", [0.0, 6.9367175234400325, -6.9367175234400325], "ur5e_45deg_rbd", False)
    #
    # if choice == 1:
    #     print(f"Choice 1: {run_choices[choice]}")
    #     api.compile_rd("ur5e_45deg_rbd", "square", [2, 1, 4], "ur5e_45deg_rd")
    #     calibrate_and_validation_plot("ur5e_45deg_rd","ur5e_45deg_calibration", "resources/Dataset/paper_plot/validation_motion.csv")
    # elif choice == 2:
    #     print(f"Choice 2: {run_choices[choice]}")
    #     api.compile_rd("ur5e_45deg_rbd", "square", [2, 1, 4], "ur5e_45deg_rd")
    #     single_prediction_plot("shoulder","ur5e_45deg_rbd","ur5e_45deg_calibration")
    # elif choice == 3:
    #     print(f"Choice 3: {run_choices[choice]}")
    #     friction_prediction_plot("shoulder")
    # elif choice == 4:
    #     print(f"Choice 4: {run_choices[choice]}")
    #     friction_prediction_plot("elbow")