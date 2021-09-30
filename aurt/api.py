import pickle
import os
import logging

from numpy.testing._private.utils import assert_equal

from aurt.rigid_body_dynamics import RigidBodyDynamics
from aurt.robot_dynamics import RobotDynamics
from aurt.robot_calibration import RobotCalibration
from aurt.robot_data import RobotData
from aurt.data_processing import *
from aurt.file_system import from_cache


def compile_rbd(mdh_path, output_path, plotting, block=True, logging_level="default"):

    if logging_level == "verbose":
        logging.basicConfig(level=logging.DEBUG)
    elif logging_level == "default":
        logging.basicConfig(level=logging.WARNING)
    l = logging.getLogger("aurt")
    
    # remove cached files that were used as intermediate results for faster computation
    # 1) regressor_joint
    rbd_regressor_joint_name = RigidBodyDynamics._filename_regressor_joint
    all_files = next(os.walk(from_cache(".")), (None, None, []))[2]
    all_files = [f for f in all_files if rbd_regressor_joint_name in f]
    for f in all_files:
        try:
            os.remove(from_cache(f))
        except:
            l.warning(f"The file: {f} could not be deleted. Please delete it manually.")
            pass
    # 2) dynamics
    rbd_file = from_cache(RigidBodyDynamics._filename + RigidBodyDynamics._file_extension)
    if os.path.isfile(rbd_file):
        try:
            os.remove(rbd_file)
        except:
            l.warning(f"The file: {rbd_file} could not be deleted. Please delete it manually.")
            pass
    # 3) regressor
    rbd_regressor_file = from_cache(RigidBodyDynamics._filename_regressor + RigidBodyDynamics._file_extension)
    if os.path.isfile(rbd_regressor_file):
        try:
            os.remove(rbd_regressor_file)
        except:
            l.warning(f"The file: {rbd_regressor_file} could not be deleted. Please delete it manually.")
            pass

    mdh = convert_file_to_mdh(mdh_path)
    rbd = RigidBodyDynamics(l, mdh)
    rbd.regressor()

    if plotting:
        rbd.plot_kinematics(block)

    # save class
    pathfile = from_cache(output_path + ".pickle")
    with open(pathfile, 'wb') as f:
        pickle.dump(rbd, f)


def compile_rd(rbd_filename, friction_torque_model, friction_viscous_powers, output_path):
    rd = RobotDynamics(rbd_filename, viscous_friction_powers=friction_viscous_powers, friction_torque_model=friction_torque_model)
    rd.regressor()

    # save class
    pathfile = from_cache(output_path + ".pickle")
    with open(pathfile, 'wb') as f:
        pickle.dump(rd, f)


def calibrate(model_path, data_path, gravity, output_params, output_calibration, plotting, logging_level="default"):

    if logging_level == "verbose":
        logging.basicConfig(level=logging.DEBUG)
    elif logging_level == "default":
        logging.basicConfig(level=logging.WARNING)
    l = logging.getLogger("aurt")

    rc = RobotCalibration(l, model_path, data_path, gravity)
    params = rc.calibrate(output_params)
    if plotting:
        rc.plot_calibration(params)

    # save class
    pathfile = from_cache(output_calibration + ".pickle")
    with open(pathfile, 'wb') as f:
        pickle.dump(rc, f)


def predict(model_path, data_path, gravity, output_path):
    filename = from_cache(model_path + ".pickle")
    with open(filename, 'rb') as f:
        rc: RobotCalibration = pickle.load(f)

    rc_predict_data = RobotData(data_path, delimiter=' ', interpolate_missing_samples=True) # TODO should we always interpolate missing samples?
    rc.predict(rc_predict_data, gravity, rc.parameters, output_path)

def calibrate_validate(model_path, data_path, gravity, calibration_data_relative, output_params, output_calibration, output_predict, plotting, logging_level="default"):

    if logging_level == "verbose":
        logging.basicConfig(level=logging.DEBUG)
    elif logging_level == "default":
        logging.basicConfig(level=logging.WARNING)
    l = logging.getLogger("aurt")

    rc = RobotCalibration(l, model_path, data_path, calibration_data_relative)
    params = rc.calibrate(output_params)
    output_pred = rc.predict(rc.robot_data_validation, gravity, params, output_predict)
    if plotting:
        rc.plot_calibrate_and_validate(params)

    # save class
    pathfile = from_cache(output_calibration + ".pickle")
    with open(pathfile, 'wb') as f:
        pickle.dump(rc, f)