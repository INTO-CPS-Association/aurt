import sympy as sp
import numpy as np
import pickle

from aurt.file_system import cache_object, from_cache
from aurt.dynamics_aux import list_2D_to_sympy_vector
from aurt.robot_dynamics import RobotDynamics
from aurt.rigid_body_dynamics import RigidBodyDynamics
from aurt.joint_dynamics import JointDynamics


class RobotDynamicsGravity(RobotDynamics):
    def __init__(self, robot_dynamics: RobotDynamics, gravity):
        # Load saved RigidBodyDynamics model
        self.rd = robot_dynamics
        self.gravity = gravity

        self.instantiate_gravity()

    def instantiate_gravity(self, gravity):

        x = 1