import os
import sys
import sympy as sp
import numpy as np
from itertools import product, chain
from multiprocessing import Pool

from aurt.file_system import cache_object, store_object, load_object
from rigid_body_dynamics import RigidBodyDynamics
from joint_dynamics import JointDynamics

# TODO: DELETE BELOW GLOBALS
from aurt.num_sym_layers import spzeros_array, spvector, npzeros_array
from aurt.globals import Njoints, get_ur_parameters_symbolic, get_ur_frames, get_ur5e_parameters
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
    def __init__(self, modified_dh, gravity=None, viscous_friction_powers=None, friction_load_model=None, hysteresis_model=None):
        self.mdh = modified_dh
        self.n_joints = RobotDynamics.number_of_joints(modified_dh)
        self.q = [sp.Integer(0)] + [sp.symbols(f"q{j}") for j in range(1, self.n_joints + 1)]
        self.qd = [sp.Integer(0)] + [sp.symbols(f"qd{j}") for j in range(1, self.n_joints + 1)]
        self.qdd = [sp.Integer(0)] + [sp.symbols(f"qdd{j}") for j in range(1, self.n_joints + 1)]
        self.tauJ = sp.symbols([f"tauJ{j}" for j in range(self.n_joints + 1)])

        self.rigid_body_dynamics = RigidBodyDynamics(modified_dh,
                                                     self.n_joints,
                                                     gravity=gravity)
        self.joint_dynamics = JointDynamics(self.n_joints,
                                            load_model=friction_load_model,
                                            hysteresis_model=hysteresis_model,
                                            viscous_powers=viscous_friction_powers)

    def __parameters_linear(self):
        """
        Returns a list of 'n_joints + 1' elements with each element comprising a list of all parameters related to that
        corresponding link and joint.
        """
        par_joint = self.joint_dynamics.parameters()
        par_rigid_body = self.rigid_body_dynamics.parameters()
        return [par_rigid_body[j] + par_joint[j] for j in range(self.n_joints + 1)]

    def __number_of_parameters_linear_model(self):
        n_par_linear = [self.rigid_body_dynamics.number_of_parameters_each_rigid_body() +
                        self.joint_dynamics.number_of_parameters_each_joint()] * (self.n_joints + 1)
        return n_par_linear

    def __regressor_linear(self):
        tau_sym_linearizable = cache_object('./dynamics_linearizable', self.__dynamics_linearizable)

        js = list(range(self.n_joints + 1))
        tau_per_task = [tau_sym_linearizable[j] for j in
                        js]  # Allows one to control how many tasks by controlling how many js
        data_per_task = list(product(zip(tau_per_task, js), [self.__parameters_linear()]))

        with Pool() as p:
            reg = p.map(self.__compute_regressor_row, data_per_task)

        return sp.Matrix(reg)

    def __regressor_linear_exist(self, regressor=None):
        # ************************* IDENTIFY (LINEAR) PARAMETERS WITH NO EFFECT ON THE DYNAMICS ************************
        # In the eq. for joint j, the dynamic parameters of proximal links (joints < j, i.e. closer to the base) will never
        # exist, i.e. the dynamic parameter of joint j will not be part of the equations for joints > j.
        if regressor is None:
            regressor = cache_object('./regressor', self.__regressor_linear)

        filename = 'parameter_indices_linear_exist'
        idx_linear_exist = cache_object(filename, lambda: self.__parameters_linear_exist(regressor)[0])

        # Removes zero columns of regressor corresponding to parameters with no influence on dynamics
        idx_linear_exist_global = np.where(list(chain.from_iterable(idx_linear_exist)))[0].tolist()
        return regressor[:, idx_linear_exist_global]

    def __compute_regressor_row(self, args):
        """
        We compute the regressor via symbolic differentiation. Each torque equation must be linearizable with respect to
        the dynamic coefficients.

        Example:
            Given the equation 'tau = sin(q)*a', tau is linearizable with respect to 'a', and the regressor 'sin(q)' can be
            obtained by partial differentiation of 'tau' with respect to 'a'.
        """
        tau_sym_linearizable_j = args[0][0]  # tau_sym_linearizable[j]
        j = args[0][1]
        p_linear = args[1]
        reg_j = sp.zeros(1, sum(self.__number_of_parameters_linear_model()))  # Initialization

        if j == 0:
            print(f"Ignore joint {j}...")
            return reg_j

        # For joint j, we loop through the parameters belonging to joints/links >= j. This is because it is physically
        # impossible for torque eq. j to include a parameter related to proximal links (< j). We divide the parameter
        # loop (for joints >= j) in two variables 'jj' and 'i':
        #   'jj' describes the joint, which the parameter belongs to
        #   'i'  describes the parameter's index/number for joint jj.
        for jj in range(j, self.n_joints + 1):  # Joint loop including this and distal (succeeding) joints (jj >= j)
            for i in range(self.__number_of_parameters_linear_model()[jj]):  # joint jj parameter loop
                column_idx = sum(self.__number_of_parameters_linear_model()[:jj]) + i
                print(f"Computing regressor(row={j}/{self.n_joints + 1}, column={column_idx}/{sum(self.__number_of_parameters_linear_model())}) by analyzing dependency of tau[{j}] on joint {jj}'s parameter {i}: {p_linear[jj][i]}")
                reg_j[0, column_idx] = sp.diff(tau_sym_linearizable_j, p_linear[jj][i])

        return reg_j

    def __dynamics_linearizable(self):

        tau_sym_rbd_linearizable = cache_object('./rigid_body_dynamics_linearizable',
                                                self.rigid_body_dynamics.get_dynamics())  #compute_tau_rbd_linearizable_parallel)

        tau_sym_jd_linearizable = cache_object(
            f'./joint_dynamics_linearizable_{self.joint_dynamics.load_model}_{self.joint_dynamics.hysteresis_model}_{self.joint_dynamics.viscous_friction_powers}', self.joint_dynamics.get_dynamics())

        return sp.Matrix(
            [tau_sym_rbd_linearizable[j] + tau_sym_jd_linearizable[j] for j in range(len(tau_sym_rbd_linearizable))])

    def __parameters_linear_exist(self, regressor=None):
        # In the regressor, identify the zero columns corresponding to parameters which does not matter to the system.
        filename_idx = 'parameter_indices_linear_exist'
        filename_npar = 'number_of_parameters_linear_exist'
        filename_par = 'parameters_linear_exist'

        if not (os.path.isfile(filename_idx) and os.path.isfile(filename_npar) and os.path.isfile(filename_par)):
            if regressor is None:
                regressor = cache_object('./regressor', self.__regressor_linear)

            idx_linear_exist = [[True for _ in range(self.__number_of_parameters_linear_model()[j])] for j in range(len(self.__parameters_linear()))]
            n_par_linear_exist = self.__number_of_parameters_linear_model().copy()
            p_linear_exist = self.__parameters_linear().copy()
            for j in range(self.n_joints + 1):
                for i in reversed(range(self.__number_of_parameters_linear_model()[j])):  # parameter loop
                    idx_column = sum(self.__number_of_parameters_linear_model()[:j]) + i
                    if not any(regressor[:, idx_column]):
                        idx_linear_exist[j][i] = False
                        n_par_linear_exist[j] -= 1
                        del p_linear_exist[j][i]

            store_object(idx_linear_exist, filename_idx)
            store_object(n_par_linear_exist, filename_npar)
            store_object(p_linear_exist, filename_par)
        else:
            idx_linear_exist = load_object(filename_idx)
            n_par_linear_exist = load_object(filename_npar)
            p_linear_exist = load_object(filename_par)

        return idx_linear_exist, n_par_linear_exist, p_linear_exist

    def __parameters_base(self):
        """Identifies the indices of the """
        # ********************************* IDENTIFY BASE PARAMETERS OF THE DYNAMICS SYSTEM ********************************
        sys.setrecursionlimit(int(1e6))  # Prevents errors in sympy lambdify
        if self.joint_dynamics.load_model.lower() == 'none':
            args_sym = self.q[1:] + self.qd[1:] + self.qdd[1:]
        else:
            args_sym = self.q[1:] + self.qd[1:] + self.qdd[1:] + self.tauJ[1:]

        regressor_with_instantiated_parameters_func = sp.lambdify(args_sym, cache_object(
            './regressor_with_instantiated_parameters',
            lambda: self.__regressor_linear_with_instantiated_parameters(robot_parameter_function=get_ur5e_parameters)))
        filename = 'parameter_indices_base'
        idx_base_global = cache_object(filename,
                                       lambda: self.__indices_base_exist(regressor_with_instantiated_parameters_func))

        idx_linear_exist, n_par_linear_exist, p_linear_exist = self.__parameters_linear_exist()
        p_linear_exist_vector = sp.Matrix(list(chain.from_iterable(p_linear_exist)))  # flatten 2D list to 1D list and convert 1D list to sympy.Matrix object

        filename = 'parameters_base_exist'
        p_base_vector = cache_object(filename, lambda: p_linear_exist_vector[idx_base_global, :])

        # Initialization
        n_par_base = n_par_linear_exist.copy()
        p_base = p_linear_exist.copy()
        idx_is_base = [[True for _ in range(len(p_base[j]))] for j in range(len(p_base))]

        print(f"p_base: {p_base}")
        for j in range(self.n_joints + 1):
            for i in reversed(range(n_par_linear_exist[j])):
                if not p_base[j][i] in p_base_vector.free_symbols:
                    idx_is_base[j][i] = False
                    n_par_base[j] -= 1
                    del p_base[j][i]
        print(f"p_base: {p_base}")

        assert np.count_nonzero(list(chain.from_iterable(idx_is_base))) == sum(n_par_base) == len(
            list(chain.from_iterable(p_base)))

        return idx_is_base, n_par_base, p_base

    def __indices_base_exist(self, regressor_with_instantiated_parameters):
        """This function computes the indices for the base parameters. The number of base parameters is obtained as the
        maximum obtainable rank of the observation matrix using a set of randomly generated dummy observations for the robot
        trajectory. The specific indices for the base parameters are obtained by conducting a QR decomposition of the
        observation matrix and analyzing the diagonal elements of the upper triangular (R) matrix."""
        # TODO: Change code such that 'dummy_pos, 'dummy_vel', 'dummy_acc', and dumm_tauJ can be called with a 2D list of
        #  indices to produce a 2D list of 'dummy_pos', 'dummy_vel', and 'dummy_acc'. This way, the regressor can be called
        #  with a 2D np.array(), which will drastically speed up the computation of the dummy observation matrix.

        # The set of dummy observations
        dummy_pos = np.array([-np.pi, -0.5 * np.pi, -0.25 * np.pi, 0.0, 0.25 * np.pi, 0.5 * np.pi, np.pi])
        dummy_vel = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
        dummy_acc = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
        dummy_tauJ = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
        assert len(dummy_pos) == len(dummy_vel) == len(dummy_acc) == len(dummy_tauJ)

        max_number_of_rank_evaluations = 100
        n_rank_evals = 0  # Loop counter for observation matrix concatenations
        n_regressor_evals_per_rank_calculation = 2  # No. of regressor evaluations per rank calculation
        rank_W = []

        if self.joint_dynamics.load_model.lower() == 'none':
            random_idx_init = [[np.random.randint(Njoints + 1) for _ in range(Njoints)] for _ in range(3)]
            dummy_args_init = np.concatenate((dummy_pos[random_idx_init[0][:]], dummy_vel[random_idx_init[1][:]],
                                              dummy_acc[random_idx_init[2][:]]))
        else:
            random_idx_init = [[np.random.randint(Njoints + 1) for _ in range(Njoints)] for _ in range(4)]
            dummy_args_init = np.concatenate((dummy_pos[random_idx_init[0][:]], dummy_vel[random_idx_init[1][:]],
                                              dummy_acc[random_idx_init[2][:]], dummy_tauJ[random_idx_init[3][:]]))

        W_N = regressor_with_instantiated_parameters(*dummy_args_init)
        W = regressor_with_instantiated_parameters(*dummy_args_init)

        # Continue the computations while the rank of the observation matrix 'W' keeps improving
        while n_rank_evals < 3 or rank_W[-1] > rank_W[-2]:  # While the rank of the observation matrix keeps increasing
            # Generate random indices for the dummy observations
            if self.joint_dynamics.load_model.lower() == 'none':
                random_idx = [[[np.random.randint(Njoints + 1) for _ in range(Njoints)]
                               for _ in range(n_regressor_evals_per_rank_calculation)] for _ in range(3)]
            else:
                random_idx = [[[np.random.randint(Njoints + 1) for _ in range(Njoints)]
                               for _ in range(n_regressor_evals_per_rank_calculation)] for _ in range(4)]

            # Evaluate the regressor in a number of dummy observations and vertically stack the regressor matrices
            for i in range(n_regressor_evals_per_rank_calculation):
                if self.joint_dynamics.load_model.lower() == 'none':
                    dummy_args = np.concatenate(
                        (dummy_pos[random_idx[0][i][:]], dummy_vel[random_idx[1][i][:]], dummy_acc[random_idx[2][i][:]]))
                else:
                    dummy_args = np.concatenate((dummy_pos[random_idx[0][i][:]], dummy_vel[random_idx[1][i][:]],
                                                 dummy_acc[random_idx[2][i][:]], dummy_tauJ[random_idx[3][i][:]]))

                reg_i = regressor_with_instantiated_parameters(*dummy_args)  # Each index contains a (Njoints x n_par) regressor matrix
                W_N = np.append(W_N, reg_i, axis=0)
            W = np.append(W, W_N, axis=0)

            # Evaluate rank of observation matrix
            rank_W.append(np.linalg.matrix_rank(W))
            print(f"Rank of observation matrix: {rank_W[-1]}")

            n_rank_evals += 1
            if n_rank_evals > max_number_of_rank_evaluations:
                raise Exception(f"Numerical estimation of the number of base inertial parameters did not converge within {n_rank_evals * n_regressor_evals_per_rank_calculation} regressor evaluations.")

        r = np.linalg.qr(W, mode='r')
        qr_numerical_threshold = 1e-12
        idx_is_base = abs(np.diag(r)) > qr_numerical_threshold
        idx_base = np.where(idx_is_base)[0].tolist()
        assert len(idx_base) == rank_W[-1]

        return idx_base

    def __regressor_linear_with_instantiated_parameters(self, robot_parameter_function):
        (_, d_num, a_num, _) = robot_parameter_function(npzeros_array)

        def to_fname(l):
            return "_".join(map(lambda s: "%1.2f" % s, l))

        data_id = f"{to_fname(d_num)}_{to_fname(a_num)}_{'%1.2f' % g_num[0]}_{'%1.2f' % g_num[1]}_{'%1.2f' % g_num[2]}"

        def load_regressor_and_subs():
            regressor_reduced = cache_object('./regressor_reduced', self.__regressor_linear_exist)
            return regressor_reduced.subs(
                RobotDynamics.sym_mat_to_subs([a, d, g, f_tcp, n_tcp], [a_num, d_num, g_num, f_tcp_num, n_tcp_num]))

        regressor_reduced_params = cache_object(f'./regressor_reduced_{data_id}', lambda: load_regressor_and_subs())

        return regressor_reduced_params

    def parameters(self):
        return self.__parameters_base()[2]

    @staticmethod
    def number_of_joints(mdh):
        return mdh.shape[0]

    @staticmethod
    def mdh_csv2obj(mdh_csv_path):
        # Converts csv file to some object, dictionary, or whatever containing the modified DH parameters. It would be
        # nice to be able to get 'a' by doing something like 'self.mdh.a', 'self.mdh['a']', or similar.
        mdh = load_object(mdh_csv_path)
        return mdh

    @staticmethod
    def sym_mat_to_subs(sym_mats, num_mats):
        subs = {}

        for s_mat, n_mat in zip(sym_mats, num_mats):
            subs = {**subs, **{s: v for s, v in zip(s_mat, n_mat) if s != 0}}

        return subs
