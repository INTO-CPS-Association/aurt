import pickle

from aurt.rigid_body_dynamics import RigidBodyDynamics
from aurt.robot_dynamics import RobotDynamics
from aurt.robot_calibration import RobotCalibration
from aurt.robot_data import RobotData
from aurt.data_processing import *
from aurt.file_system import from_cache


def compile_rbd(mdh_path, gravity, output_path, plotting, block=True):
    mdh = convert_file_to_mdh(mdh_path)
    rbd = RigidBodyDynamics(mdh, gravity)
    rbd.regressor()

    if plotting:
        rbd.plot_kinematics(block)

    # save class
    pathfile = from_cache(output_path + ".pickle")
    with open(pathfile, 'wb') as f:
        pickle.dump(rbd, f)

def compile_rd(rbd_filename, friction_load_model, friction_viscous_powers, output_path):
    rd = RobotDynamics(rbd_filename, viscous_friction_powers=friction_viscous_powers, friction_load_model=friction_load_model)
    rd.regressor()

    # save class
    pathfile = from_cache(output_path + ".pickle")
    with open(pathfile, 'wb') as f:
        pickle.dump(rd, f)


def calibrate(model_path, data_path, output_params, output_calibration, plotting):
    rc = RobotCalibration(model_path, data_path)
    params = rc.calibrate(output_params)
    if plotting:
        rc.plot_calibration(params)

    # save class
    pathfile = from_cache(output_calibration + ".pickle")
    with open(pathfile, 'wb') as f:
        pickle.dump(rc, f)


def predict(model_path, data_path, output_path):
    filename = from_cache(model_path + ".pickle")
    with open(filename, 'rb') as f:
        rc: RobotCalibration = pickle.load(f)

    rc_predict_data = RobotData(data_path, delimiter=' ', interpolate_missing_samples=True) # TODO should we always interpolate missing samples?
    rc.predict(rc_predict_data, rc.parameters, output_path)



def calibrate_validate(model_path, data_path, calibration_data_relative, output_params, output_calibration, output_predict, plotting):
    rc = RobotCalibration(model_path, data_path, calibration_data_relative)
    params = rc.calibrate(output_params)
    output_pred = rc.predict(rc.robot_data_validation, params, output_predict)
    if plotting:
        rc.plot_calibrate_and_validate(params)

    # save class
    pathfile = from_cache(output_calibration + ".pickle")
    with open(pathfile, 'wb') as f:
        pickle.dump(rc, f)