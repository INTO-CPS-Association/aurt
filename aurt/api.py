import pickle
import logging
import numpy as np

from aurt.caching import Cache, clear_cache_dir, PersistentPickleCache
from aurt.rigid_body_dynamics import RigidBodyDynamics
from aurt.robot_dynamics import RobotDynamics
from aurt.robot_calibration import RobotCalibration
from aurt.robot_data import RobotData
from aurt.data_processing import *
from aurt.file_system import store_csv


def compile_rbd(mdh_path, output_path, plotting, cache: PersistentPickleCache, block=True):
    l = logging.getLogger("aurt")

    l.info(f"Clearing cache {cache._base_directory}.")
    clear_cache_dir(cache._base_directory)

    mdh = convert_file_to_mdh(mdh_path)
    rbd = RigidBodyDynamics(l, mdh, cache)
    rbd.regressor()

    if plotting:
        rbd.plot_kinematics(block)

    # save class
    with open(output_path, 'wb') as f:
        pickle.dump(rbd, f)


def compile_rd(rbd_filename, friction_torque_model, friction_viscous_powers, output_path, cache: Cache):
    l = logging.getLogger("aurt")

    # Load RigidBodyDynamics
    with open(rbd_filename, 'rb') as f:
        rigid_body_dynamics: RigidBodyDynamics = pickle.load(f)

    rd = RobotDynamics(rigid_body_dynamics, l, cache, viscous_friction_powers=friction_viscous_powers, friction_torque_model=friction_torque_model)
    rd.regressor()

    # save class
    with open(output_path, 'wb') as f:
        pickle.dump(rd, f)


def calibrate(model_path, data_path, gravity, output_params, output_calibration, plotting):
    l = logging.getLogger("aurt")
    gravity = np.array(gravity)

    # Load RobotDynamics
    with open(model_path, 'rb') as f:
        robot_dynamics: RobotDynamics = pickle.load(f)

    rc = RobotCalibration(l, robot_dynamics, data_path, gravity)
    params = rc.calibrate()

    if plotting:
        rc.plot_calibration(params)

    # save class and parameters
    store_csv(output_params, params)
    with open(output_calibration, 'wb') as f:
        pickle.dump(rc, f)


def predict(model_path, data_path, gravity, output_path):
    l = logging.getLogger("aurt")
    gravity = np.array(gravity)

    with open(model_path, 'rb') as f:
        rc: RobotCalibration = pickle.load(f)

    rc_predict_data = RobotData(l, data_path, delimiter=' ', interpolate_missing_samples=True) # TODO should we always interpolate missing samples?
    prediction = rc.predict(rc_predict_data, gravity, rc.parameters)

    # Store CSV
    store_csv(output_path, prediction)


def calibrate_validate(model_path, data_path, gravity, calibration_data_relative, output_params, output_calibration, output_predict, plotting):
    l = logging.getLogger("aurt")
    gravity = np.array(gravity)

    # Load RobotDynamics
    with open(model_path, 'rb') as f:
        robot_dynamics: RobotDynamics = pickle.load(f)

    rc = RobotCalibration(l, robot_dynamics, data_path, gravity, calibration_data_relative)
    params = rc.calibrate()
    output_pred = rc.predict(rc.robot_data_validation, gravity, params)

    if plotting:
        rc.plot_calibrate_and_validate(params)

    # save class, parameters, and prediction
    store_csv(output_params, params)
    store_csv(output_predict, output_pred)
    with open(output_calibration, 'wb') as f:
        pickle.dump(rc, f)
