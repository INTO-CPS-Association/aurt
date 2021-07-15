import sympy as sp
import numpy as np
import pickle

from aurt.file_system import cache_object, from_cache
from aurt.rigid_body_dynamics import RigidBodyDynamics
from aurt.joint_dynamics import JointDynamics


class RobotDynamics:
    def __init__(self, rbd_filename, tcp_force_torque=None, viscous_friction_powers=None, friction_load_model=None, hysteresis_model=None): # TODO ask Emil about tcp_force_torque
        # Load saved RigidBodyDynamics model
        filename = from_cache(rbd_filename + ".pickle")
        with open(filename, 'rb') as f:
            self.rigid_body_dynamics: RigidBodyDynamics = pickle.load(f)
        self.n_joints = self.rigid_body_dynamics.mdh.n_joints
        self.q = [sp.Integer(0)] + [sp.symbols(f"q{j}") for j in range(1, self.n_joints + 1)]
        self.qd = [sp.Integer(0)] + [sp.symbols(f"qd{j}") for j in range(1, self.n_joints + 1)]
        self.qdd = [sp.Integer(0)] + [sp.symbols(f"qdd{j}") for j in range(1, self.n_joints + 1)]
        self.tauJ = sp.symbols([f"tauJ{j}" for j in range(self.n_joints + 1)])

        self.joint_dynamics = JointDynamics(self.n_joints,
                                            load_model=friction_load_model,
                                            hysteresis_model=hysteresis_model,
                                            viscous_powers=viscous_friction_powers)

    def filepath_regressor_joint(self, j):
        return from_cache(f'robot_dynamics_regressor_joint_{j}')

    def parameters(self):
        """
        Returns a list of 'n_joints + 1' elements with each element comprising a list of all parameters related to that
        corresponding link and joint.
        """
        par_rigid_body = self.rigid_body_dynamics.params
        par_joint = self.joint_dynamics.parameters()
        return [par_rigid_body[j] + par_joint[j] for j in range(self.n_joints)]

    def number_of_parameters(self):
        n_par_rbd = self.rigid_body_dynamics.n_params
        n_par_jd = self.joint_dynamics.number_of_parameters()
        return [n_par_rbd[j] + n_par_jd[j] for j in range(self.n_joints)]

    def observation_matrix_joint(self, j, q_num, qd_num, qdd_num):
        assert q_num.shape == qd_num.shape == qdd_num.shape
        assert 0 <= j < self.n_joints

        n_samples = q_num.shape[1]
        obs_mat_j = np.empty((n_samples, 0))
        obs_mat_j_rbd = self.rigid_body_dynamics.observation_matrix_joint(j, q_num, qd_num, qdd_num)
        for j_par in range(self.n_joints):
            if j_par == j:
                tauJ_j_basis_num = np.sum(obs_mat_j_rbd, axis=1).transpose()
                obs_mat_j_jd = self.joint_dynamics.observation_matrix_joint(j, qd_num[j, :], tauJ_j_basis_num)
            else:
                obs_mat_j_jd = np.zeros((n_samples, self.joint_dynamics.number_of_parameters()[j_par]))

            col_start = sum(self.rigid_body_dynamics.n_params[:j_par])
            col_end = col_start + self.rigid_body_dynamics.n_params[j_par]
            obs_mat_j_jpar_rbd = obs_mat_j_rbd[:, col_start:col_end]
            obs_mat_j = np.hstack((obs_mat_j, obs_mat_j_jpar_rbd, obs_mat_j_jd))
        return obs_mat_j

    def observation_matrix(self, q_num, qd_num, qdd_num):
        assert q_num.shape == qd_num.shape == qdd_num.shape

        n_samples = q_num.shape[1]

        # Merge observation matrices for rigid-body dynamics and joint dynamcis
        observation_matrix = np.empty((self.n_joints*n_samples, sum(self.number_of_parameters())))
        for j in range(self.n_joints):
            observation_matrix[j*n_samples:(j+1)*n_samples, :] = self.observation_matrix_joint(j, q_num, qd_num, qdd_num)

        return observation_matrix

    def regressor_joint(self, j):
        """Constructs row 'j' of the regressor matrix."""
        def compute_regressor_joint():
            reg_j = sp.Matrix(1, 0, [])

            for j_par in range(self.n_joints):
                reg_j_rbd_par_j = self.rigid_body_dynamics.regressor_joint_parameters_for_joint(j, j_par)
                if j_par == j:
                    reg_j_jd = self.joint_dynamics.regressor()[j_par]
                else:
                    reg_j_jd = sp.zeros(1, self.joint_dynamics.number_of_parameters()[j_par])

                reg_j = sp.Matrix.hstack(reg_j, reg_j_rbd_par_j, reg_j_jd)
            return reg_j

        return compute_regressor_joint()#cache_object(self.filepath_regressor_joint(j+1), compute_regressor_joint)

    def regressor(self, output_filename="robot_dynamics_regressor"):
        filepath_regressor = from_cache(output_filename)

        def compute_regressor():
            """Merges regressor matrices for rigid-body dynamics and joint dynamics to construct a regressor for the
            robot dynamics."""
            reg = sp.zeros(self.n_joints, sum(self.number_of_parameters()))
            for j in range(self.n_joints):
                reg[j, :] = self.regressor_joint(j)

            # for j in range(self.n_joints):
            #     cache_object(from_cache(f"robot_dynamics_regressor_joint_{j+1}"), lambda: reg[j, :])
            return reg

        return cache_object(filepath_regressor, compute_regressor)
