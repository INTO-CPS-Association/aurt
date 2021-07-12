from aurt.rigid_body_dynamics import RigidBodyDynamics
from aurt.data_processing import *


def compile_rbd(mdh_path, gravity, output_path):
    mdh = convert_file_to_mdh(mdh_path)
    rbd = RigidBodyDynamics(mdh, gravity)
    rbd.regressor(output_path)
    

def compile_jointd(model_rbd_path, friction_load_model, friction_viscous_powers, friction_hysteresis_model, output_path):
    pass

def calibrate(model_path, data_path, reduced, output_path):
    pass

