import os
import sys
import sympy as sp
import numpy as np

from aurt.file_system import cache_object, store_object, load_object, from_cache
from aurt.rigid_body_dynamics import RigidBodyDynamics
from aurt.joint_dynamics import JointDynamics

# TODO: DELETE BELOW GLOBALS
from aurt.num_sym_layers import spzeros_array
from aurt.globals import get_ur_parameters_symbolic
gx, gy, gz = sp.symbols(f"gx gy gz")
g = sp.Matrix([gx, gy, gz])
g_num = np.array([0.0, 0.0, 9.81])

fx, fy, fz, nx, ny, nz = sp.symbols(f"fx fy fz nx ny nz")
f_tcp = sp.Matrix([fx, fy, fz])  # Force at the TCP
n_tcp = sp.Matrix([nx, ny, nz])  # Moment at the TCP
f_tcp_num = sp.Matrix([0.0, 0.0, 0.0])
n_tcp_num = sp.Matrix([0.0, 0.0, 0.0])

(_, d, a, _) = get_ur_parameters_symbolic(spzeros_array)


class RobotDynamics:
    def __init__(self, modified_dh, gravity=None, tcp_force_torque=None, viscous_friction_powers=None, friction_load_model=None, hysteresis_model=None):
        print(f"RobotDynamics.__init__()")
        # TODO: Change constructor to take directly 'RigidBodyDynamics' and 'JointDynamics' objects instead of their
        #  arguments.

        self.mdh = modified_dh
        self.n_joints = RobotDynamics.number_of_joints(modified_dh)  # TODO: Correct this
        self.q = [sp.Integer(0)] + [sp.symbols(f"q{j}") for j in range(1, self.n_joints + 1)]
        self.qd = [sp.Integer(0)] + [sp.symbols(f"qd{j}") for j in range(1, self.n_joints + 1)]
        self.qdd = [sp.Integer(0)] + [sp.symbols(f"qdd{j}") for j in range(1, self.n_joints + 1)]
        self.tauJ = sp.symbols([f"tauJ{j}" for j in range(self.n_joints + 1)])

        self.rigid_body_dynamics = RigidBodyDynamics(modified_dh,
                                                     self.n_joints,
                                                     gravity=gravity,
                                                     tcp_force_torque=tcp_force_torque)
        self.joint_dynamics = JointDynamics(self.n_joints,
                                            load_model=friction_load_model,
                                            hysteresis_model=hysteresis_model,
                                            viscous_powers=viscous_friction_powers)

        self.filepath_regressor = from_cache('robot_dynamics_regressor')
        self.__filepath_regressor_joint = from_cache('robot_dynamics_regressor_joint_')

    def filepath_regressor_joint(self, j):
        return from_cache(f'{self.__filepath_regressor_joint}{j}')

    def parameters(self):
        """
        Returns a list of 'n_joints + 1' elements with each element comprising a list of all parameters related to that
        corresponding link and joint.
        """
        print(f"RobotDynamics.__parameters_linear()")

        par_rigid_body = self.rigid_body_dynamics.parameters()
        par_joint = self.joint_dynamics.parameters()
        return [par_rigid_body[j] + par_joint[j] for j in range(self.n_joints + 1)]

    def number_of_parameters(self):
        n_par_rbd = self.rigid_body_dynamics.number_of_parameters()
        n_par_jd = self.joint_dynamics.number_of_parameters()
        return [n_par_rbd[j] + n_par_jd[j] for j in range(self.n_joints + 1)]

    def observation_matrix_joint(self, j, q_num, qd_num, qdd_num):
        assert q_num.shape == qd_num.shape == qdd_num.shape
        assert 0 < j <= self.n_joints

        obs_mat_j_rbd = self.rigid_body_dynamics.observation_matrix_joint(j+1, q_num, qd_num, qdd_num)
        tauJ_j_basis_num = np.sum(obs_mat_j_rbd, axis=1).transpose()
        obs_mat_j_jd = self.joint_dynamics.observation_matrix_joint(j+1, qd_num[j, :], tauJ_j_basis_num)

        n_samples = q_num.shape[1]
        obs_mat_j = np.empty((n_samples, sum(self.number_of_parameters())))  #obs_mat_j_rbd  # Initialization
        column_idx_split = [sum(self.number_of_parameters()[:j]) for j in range(1, self.n_joints + 1)]
        obs_mat_j_rbd_split = np.hsplit(obs_mat_j_rbd, column_idx_split)
        for j in reversed(range(self.n_joints)):
            f=1
            # obs_mat_j[] = np.hstack((obs_mat_j_rbd_split[j], obs_mat_j_jd))
            # obs_mat_j[] = #np.insert(obs_mat, obs_mat_j_jd, column_idx, axis=1)  #= np.concatenate(obs_mat_j_rbd[column_idx:column_idx], obs_mat_j_jd)

        # return obs_mat

    def observation_matrix(self, q_num, qd_num, qdd_num):
        assert q_num.shape == qd_num.shape == qdd_num.shape

        n_par_rbd = self.rigid_body_dynamics.number_of_parameters()
        n_par_jd = self.joint_dynamics.number_of_parameters()

        n_samples = q_num.shape[1]

        # Merge observation matrices for rigid-body dynamics and joint dynamcis
        observation_matrix = np.empty((self.n_joints*n_samples, sum(self.number_of_parameters())))
        for j in reversed(range(self.n_joints + 1)):
            column_idx_start_rbd = sum(n_par_rbd[:j])
            column_idx_end_rbd = column_idx_start_rbd + n_par_rbd[j]
            column_idx_start_jd = sum(n_par_jd[:j])
            column_idx_end_jd = column_idx_start_jd + n_par_jd[j]

            obs_mat_j_rbd = self.rigid_body_dynamics.observation_matrix_joint(j+1, q_num, qd_num, qdd_num)
            tauJ_j_basis_num = np.sum(obs_mat_j_rbd, axis=1).transpose()
            obs_mat_j_jd = self.joint_dynamics.observation_matrix_joint(j+1, qd_num[j, :], tauJ_j_basis_num)

            observation_matrix = observation_matrix.row_stack(obs_mat_j_rbd).row_stack(obs_mat_j_jd)

        for j in range(self.n_joints):
            obs_mat_j_rbd = self.rigid_body_dynamics.observation_matrix_joint(j+1, q_num, qd_num, qdd_num)
            tauJ_j_basis_num = np.sum(obs_mat_j_rbd, axis=1).transpose()
            obs_mat_j_jd = self.joint_dynamics.observation_matrix_joint(j+1, qd_num[j, :], tauJ_j_basis_num)
            observation_matrix[j*n_samples:(j+1)*n_samples, :] = np.concatenate(obs_mat_j_rbd, obs_mat_j_jd)

        return observation_matrix

    def regressor_joint(self, j):
        return cache_object(self.filepath_regressor_joint(j), lambda: self.regressor()[j, :])

    def regressor(self):
        def compute_regressor():
            reg_rbd = self.rigid_body_dynamics.regressor()
            reg_jd = self.joint_dynamics.regressor()
            n_par_rbd = self.rigid_body_dynamics.number_of_parameters()
            n_par_jd = self.joint_dynamics.number_of_parameters()

            # Merge regressors for rigid-body dynamics and joint dynamcis
            # reg = []
            # reg_matrix = sp.zeros((self.n_joints, sum(self.number_of_parameters())))
            reg_matrix = sp.Matrix(0, self.n_joints, [])
            for j in range(self.n_joints + 1):
                column_idx_start_rbd = sum(n_par_rbd[:j])
                column_idx_end_rbd = column_idx_start_rbd + n_par_rbd[j]
                column_idx_start_jd = sum(n_par_jd[:j])
                column_idx_end_jd = column_idx_start_jd + n_par_jd[j]
                reg_rbd_j = reg_rbd[:, column_idx_start_rbd:column_idx_end_rbd]
                reg_jd_j = reg_jd[:, column_idx_start_jd:column_idx_end_jd]
                # reg[j] = reg_rbd[:, column_idx_start_rbd:column_idx_end_rbd]
                # reg_matrix = reg_rbd[:, column_idx_start_rbd:column_idx_end_rbd].row_join(reg_jd[:, column_idx_start_jd:column_idx_end_jd])
                reg_matrix = reg_matrix.row_join(reg_rbd_j).row_join(reg_jd_j)

            for j in range(self.n_joints):
                cache_object(from_cache(f"robot_dynamics_regressor_joint_{j}"), lambda: reg_matrix[j, :])

        return cache_object(from_cache('robot_dynamics_regressor'), compute_regressor)

    @staticmethod
    def number_of_joints(mdh):
        return 6#mdh.shape[0]  # TODO: Fix the 'number_of_joints' function
