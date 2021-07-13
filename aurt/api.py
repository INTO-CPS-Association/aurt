from aurt.rigid_body_dynamics import RigidBodyDynamics
from aurt.robot_dynamics import RobotDynamics
from aurt.data_processing import *

from aurt.file_system import from_cache
import pickle

def compile_rbd(mdh_path, gravity, output_path):
    mdh = convert_file_to_mdh(mdh_path)
    print(f"mdh: {mdh}") # TODO remove
    rbd = RigidBodyDynamics(mdh, gravity)
    rbd.regressor()

    # save class
    pathfile = from_cache(output_path + ".pickle")
    with open(pathfile, 'wb') as f:
        pickle.dump(rbd, f)

def compile_rd(rbd_filename, friction_load_model, friction_viscous_powers, friction_hysteresis_model, output_path):
    rd = RobotDynamics(rbd_filename, viscous_friction_powers=friction_viscous_powers, friction_load_model=friction_load_model, hysteresis_model=friction_hysteresis_model)
    rd.regressor(output_path)

def calibrate(model_path, data_path, reduced, output_path):
    pass

