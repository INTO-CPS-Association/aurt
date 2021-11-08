import sympy as sp
import numpy as np
from logging import Logger

from aurt.caching import Cache
from aurt.dynamics_aux import list_2D_to_sympy_vector, number_of_elements_in_nested_list
from aurt.rigid_body_dynamics import RigidBodyDynamics
from aurt.joint_dynamics import JointDynamics
from aurt.linear_system import LinearSystem


class RobotDynamics(LinearSystem):
    def __init__(self, rigid_body_dynamics: RigidBodyDynamics, logger: Logger, cache: Cache, viscous_friction_powers=None, friction_torque_model=None, hysteresis_model=None):
        super().__init__(logger, cache, name="robot dynamics")

        self.rigid_body_dynamics = rigid_body_dynamics
        self.n_joints = self.rigid_body_dynamics.mdh.n_joints
        self.q = [sp.Integer(0)] + [sp.symbols(f"q{j}") for j in range(1, self.n_joints + 1)]
        self.qd = [sp.Integer(0)] + [sp.symbols(f"qd{j}") for j in range(1, self.n_joints + 1)]
        self.qdd = [sp.Integer(0)] + [sp.symbols(f"qdd{j}") for j in range(1, self.n_joints + 1)]
        self.tauJ = sp.symbols([f"tauJ{j}" for j in range(self.n_joints + 1)])

        self.joint_dynamics = JointDynamics(logger,
                                            cache,
                                            self.n_joints,
                                            load_model=friction_torque_model,
                                            hysteresis_model=hysteresis_model,
                                            viscous_powers=viscous_friction_powers)

        super().compute_linearly_independent_system()

    def states(self):
        states_jd = self.joint_dynamics.states()
        states_rbd = self.rigid_body_dynamics.states()
        states_rd = states_rbd
        for j in range(self.n_joints):
            states_rd[j] += states_jd[j]
        states_rd_uniques = [list(dict.fromkeys(elem)) for elem in states_rd]  # eliminate non-uniques
        return states_rd_uniques

    def _parameters_full(self):
        """
        Returns a list of 'n_joints' elements with each element 'j' = 1, ..., 'n_joints' comprising a list of
        Base Parameters (BP) related to link and joint 'j'.
        """
        par_rigid_body = self.rigid_body_dynamics._parameters_full()
        par_joint = self.joint_dynamics._parameters_full()
        return [par_rigid_body[j] + par_joint[j] for j in range(self.n_joints)]

    def parameters_essential(self, q_num, qd_num, qdd_num, g_num):
        # 1. Do SVD of observation matrix
        # 2. Choose Essential Inertial Parameters (EIP) based on some criteria (Akaike's Information Criteria, i.e. AIC?)
        # ...
        # obs_mat = self.observation_matrix(q_num, qd_num, qdd_num, g_num)
        # s = np.linalg.svd(obs_mat)
        return 1

    def observation_matrix_joint(self, j, states_num):
        """
        Constructs the observation matrix for joint 'j' by evaluating the regressor for joint 'j'
        (the j'th row of the regressor matrix) in the provided data. The data should consist of a
        numpy.array with states (see the method states() for a description hereof) along axis 0 
        and time along axis 1.
        """

        assert 0 <= j < self.n_joints
        assert states_num.shape[0] == number_of_elements_in_nested_list(self.rigid_body_dynamics.states())

        n_samples = states_num.shape[1]

        n_par_rbd = self.rigid_body_dynamics.number_of_parameters()
        n_par_jd = self.joint_dynamics.number_of_parameters()
        n_par_rd = self.number_of_parameters()

        obs_mat_j = np.zeros((n_samples, sum(n_par_rd)))
        obs_mat_j_rbd = self.rigid_body_dynamics.observation_matrix_joint(j, states_num)
        tauJ_basis_num = np.zeros((self.n_joints, n_samples))
        tauJ_basis_num[j, :] = np.sum(obs_mat_j_rbd, axis=1).T

        qd_num = states_num[1::3, :]
        states_jd_num = np.empty((qd_num.shape[0] + tauJ_basis_num.shape[0], qd_num.shape[1]))
        states_jd_num[0::2, :] = qd_num
        states_jd_num[1::2, :] = tauJ_basis_num

        for par_j in range(self.n_joints):
            obs_mat_rbd_j_parj = self.rigid_body_dynamics.observation_matrix_joint_parameters_for_joint(j, par_j, states_num)
            obs_mat_jd_j_parj = self.joint_dynamics.observation_matrix_joint_parameters_for_joint(j, par_j, states_jd_num)

            col_start_rbd = sum(n_par_rbd[:par_j]) + sum(n_par_jd[:par_j])
            col_end_rbd = col_start_rbd + n_par_rbd[par_j]
            col_start_jd = col_end_rbd
            col_end_jd = col_start_jd + n_par_jd[par_j]

            obs_mat_j[:, col_start_rbd:col_end_rbd] = obs_mat_rbd_j_parj
            obs_mat_j[:, col_start_jd:col_end_jd] = obs_mat_jd_j_parj

        return obs_mat_j

    def observation_matrix_joint_parameters_for_joint():
        """
        The observation matrix is implemented in 'observation_matrix_joint', 
        thus this function is not needed and is therefore disabled.
        """
        pass

    def _regressor_joint_parameters_for_joint(self, j, par_j):
        reg_rbd_j_par_j = self.rigid_body_dynamics._regressor_joint_parameters_for_joint(j, par_j)
        reg_jd_j_par_j = self.joint_dynamics._regressor_joint_parameters_for_joint(j, par_j)
        return sp.Matrix.hstack(reg_rbd_j_par_j, reg_jd_j_par_j)

    def dynamics(self):
        """The robot dynamics consisting of; 1) the rigid-body dynamics formulated in terms of
        the Base Inertial Parameters (BIP) and 2) the joint dynamics."""
        return self.regressor() @ list_2D_to_sympy_vector(self.parameters)
