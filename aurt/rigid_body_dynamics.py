import numpy as np
import sympy as sp
from multiprocessing import Pool
from itertools import product, chain
import sys

from aurt.file_system import cache_object, from_cache
from aurt.torques import compute_torques_symbolic_ur
from aurt.dynamics_aux import sym_mat_to_subs, replace_first_moments, compute_regressor_row


# TODO: DELETE BELOW GLOBALS
from aurt.num_sym_layers import spzeros_array, spvector, npzeros_array
from aurt.globals import get_ur_parameters_symbolic, get_ur_frames, get_ur5e_parameters

(_, d, a, _) = get_ur_parameters_symbolic(spzeros_array)


class RigidBodyDynamics:
    __qr_numerical_threshold = 1e-12  # Numerical threshold used in identifying the base inertial parameters
    __max_number_of_rank_evaluations = 100  # Max. no. of rank evaluations in computing the base inertial parameters
    __n_regressor_evals_per_rank_calculation = 2  # No. of regressor evaluations per rank calculation

    def __init__(self, modified_dh, n_joints, gravity=None, tcp_force_torque=None):
        self.mdh = modified_dh
        self.n_joints = n_joints

        # TODO: Change such that q, qd, and qdd are generated based on 'mdh'
        self.q = [sp.Integer(0)] + [sp.symbols(f"q{j}") for j in range(1, self.n_joints + 1)]
        self.qd = [sp.Integer(0)] + [sp.symbols(f"qd{j}") for j in range(1, self.n_joints + 1)]
        self.qdd = [sp.Integer(0)] + [sp.symbols(f"qdd{j}") for j in range(1, self.n_joints + 1)]

        self.__m = sp.symbols([f"m{j}" for j in range(self.n_joints + 1)])
        self.__mX = sp.symbols([f"mX{j}" for j in range(self.n_joints + 1)])
        self.__mY = sp.symbols([f"mY{j}" for j in range(self.n_joints + 1)])
        self.__mZ = sp.symbols([f"mZ{j}" for j in range(self.n_joints + 1)])
        self.__pc = get_ur_frames(None, spvector)
        self.__m_pc = [[self.__mX[j], self.__mY[j], self.__mZ[j]] for j in range(self.n_joints + 1)]

        self.__XX = sp.symbols([f"XX{j}" for j in range(self.n_joints + 1)])
        self.__XY = sp.symbols([f"XY{j}" for j in range(self.n_joints + 1)])
        self.__XZ = sp.symbols([f"XZ{j}" for j in range(self.n_joints + 1)])
        self.__YY = sp.symbols([f"YY{j}" for j in range(self.n_joints + 1)])
        self.__YZ = sp.symbols([f"YZ{j}" for j in range(self.n_joints + 1)])
        self.__ZZ = sp.symbols([f"ZZ{j}" for j in range(self.n_joints + 1)])
        self.__i_cor = [sp.zeros(3, 3) for _ in range(self.n_joints + 1)]
        for j in range(self.n_joints + 1):
            self.__i_cor[j] = sp.Matrix([
                [self.__XX[j], self.__XY[j], self.__XZ[j]],
                [self.__XY[j], self.__YY[j], self.__YZ[j]],
                [self.__XZ[j], self.__YZ[j], self.__ZZ[j]]
            ])

        if gravity is None:
            print(f"No gravity direction was specified. Assuming the first joint axis of rotation to be parallel to gravity...")
            gravity = [0, 0, -9.81]
        self.gravity = gravity
        gx, gy, gz = sp.symbols(f"gx gy gz")
        self.__g = sp.Matrix([gx, gy, gz])

        fx, fy, fz, nx, ny, nz = sp.symbols(f"fx fy fz nx ny nz")
        self.__f_tcp = sp.Matrix([fx, fy, fz])  # Force at the TCP
        self.__n_tcp = sp.Matrix([nx, ny, nz])  # Moment at the TCP

        if tcp_force_torque is None:
            f_tcp_num = np.array([0, 0, 0])
            n_tcp_num = np.array([0, 0, 0])
            tcp_force_torque = [f_tcp_num, n_tcp_num]
        self.__f_tcp_num = tcp_force_torque[0]
        self.__n_tcp_num = tcp_force_torque[1]

        # Filepaths
        self.filepath_dynamics = from_cache('rigid_body_dynamics')
        self.filepath_regressor = from_cache('rigid_body_dynamics_regressor')
        self.__filepath_regressor_joint = from_cache('rigid_body_dynamics_regressor_joint_')

    def filepath_regressor_joint(self, j):
        return from_cache(f'{self.__filepath_regressor_joint}{j}')

    def parameters(self):
        """
        Returns a list of 'n_joints + 1' elements with each element comprising a list of all rigid body parameters
        related to that corresponding link.
        """

        return self.__base_parameters_information()[2]

    def number_of_parameters(self):
        return self.__base_parameters_information()[1]

    def __parameters_linear(self):
        """
        Returns a list of 'n_joints + 1' elements with each element comprising a list of all parameters related to that
        corresponding rigid body.
        """
        return [[self.__XX[j], self.__XY[j], self.__XZ[j], self.__YY[j], self.__YZ[j], self.__ZZ[j],
                 self.__mX[j], self.__mY[j], self.__mZ[j], self.__m[j]] for j in range(self.n_joints + 1)]

    def __regressor_linear(self):
        dynamics_linearizable = self.dynamics()

        js = list(range(self.n_joints + 1))
        tau_per_task = [dynamics_linearizable[j] for j in js]  # Allows one to control how many tasks by controlling how many js
        data_per_task = list(product(zip(tau_per_task, js), [self.n_joints], [self.__parameters_linear()]))

        with Pool() as p:
            reg = p.map(compute_regressor_row, data_per_task)

        return sp.Matrix(reg)

    def __regressor_linear_exist(self):
        # In the eq. for joint j, the dynamic parameters of proximal links (joints < j, i.e. closer to the base) will never
        # exist, i.e. the dynamic parameter of joint j will not be part of the equations for joints > j.
        regressor = self.__regressor_linear()
        idx_linear_exist = self.__parameters_linear_exist(regressor)[0]

        # Removes zero columns of regressor corresponding to parameters with no influence on dynamics
        idx_linear_exist_global = np.where(list(chain.from_iterable(idx_linear_exist)))[0].tolist()

        return regressor[:, idx_linear_exist_global]

    def __parameters_linear_exist(self, regressor):
        # In the regressor, identify the zero columns corresponding to parameters which does not matter to the system.
        number_of_parameters_linear_model = [len(self.__parameters_linear()[j]) for j in range(len(self.__parameters_linear()))]

        # Initialization
        idx_linear_exist = [[True for _ in range(number_of_parameters_linear_model[j])] for j in range(len(self.__parameters_linear()))]
        n_par_linear_exist = number_of_parameters_linear_model.copy()
        p_linear_exist = self.__parameters_linear().copy()

        # Computing parameter existence
        for j in range(self.n_joints + 1):
            for i in reversed(range(number_of_parameters_linear_model[j])):  # parameter loop
                idx_column = sum(number_of_parameters_linear_model[:j]) + i
                if not any(regressor[:, idx_column]):
                    idx_linear_exist[j][i] = False
                    n_par_linear_exist[j] -= 1
                    del p_linear_exist[j][i]

        return idx_linear_exist, n_par_linear_exist, p_linear_exist

    def __base_parameters_information(self):
        """Identifies the base parameters of the robot dynamics."""
        def compute_base_parameters_information():
            args_sym = self.q[1:] + self.qd[1:] + self.qdd[1:]
            sys.setrecursionlimit(int(1e6))  # Prevents errors in sympy lambdify
            regressor_with_instantiated_parameters_func = sp.lambdify(args_sym, cache_object(
                from_cache('rigid_body_dynamics_regressor_with_instantiated_parameters'),
                lambda: self.__regressor_linear_with_instantiated_parameters(robot_parameter_function=get_ur5e_parameters)))
            idx_base_global = self.__indices_base_exist(regressor_with_instantiated_parameters_func)

            idx_linear_exist, n_par_linear_exist, p_linear_exist = self.__parameters_linear_exist(self.__regressor_linear())
            p_linear_exist_vector = sp.Matrix(list(chain.from_iterable(p_linear_exist)))  # flatten 2D list to 1D list and convert 1D list to sympy.Matrix object

            p_base_vector = cache_object(from_cache('rigid_body_dynamics_base_parameters_vector'), lambda: p_linear_exist_vector[idx_base_global, :])

            # Initialization
            n_par_base = n_par_linear_exist.copy()
            p_base = p_linear_exist.copy()
            idx_is_base = [[True for _ in range(len(p_base[j]))] for j in range(len(p_base))]

            for j in range(self.n_joints + 1):
                for i in reversed(range(n_par_linear_exist[j])):
                    if not p_base[j][i] in p_base_vector.free_symbols:
                        idx_is_base[j][i] = False
                        n_par_base[j] -= 1
                        del p_base[j][i]

            assert np.count_nonzero(list(chain.from_iterable(idx_is_base))) == sum(n_par_base) == len(
                list(chain.from_iterable(p_base)))

            return idx_is_base, n_par_base, p_base

        idx_is_base, n_par_base, p_base = cache_object(from_cache('rigid_body_dynamics_parameter_information'),
                                                       compute_base_parameters_information)

        return idx_is_base, n_par_base, p_base

    def __regressor_linear_with_instantiated_parameters(self, robot_parameter_function):
        (_, d_num, a_num, _) = robot_parameter_function(npzeros_array)

        # def to_fname(l):
        #     return "_".join(map(lambda s: "%1.2f" % s, l))

        # data_id = f"{to_fname(d_num)}_{to_fname(a_num)}_{'%1.2f' % g_num[0]}_{'%1.2f' % g_num[1]}_{'%1.2f' % g_num[2]}"

        def load_regressor_and_subs():
            regressor_reduced = self.__regressor_linear_exist()
            return regressor_reduced.subs(
                sym_mat_to_subs([a, d, self.__g, self.__f_tcp, self.__n_tcp], [a_num, d_num, self.gravity, self.__f_tcp_num, self.__n_tcp_num]))

        # regressor_reduced_params = cache_object(from_cache(f'rigid_body_dynamics_regressor_linear_exist_{data_id}'),
        #                                         lambda: load_regressor_and_subs())

        return load_regressor_and_subs()

    def __indices_base_exist(self, regressor_with_instantiated_parameters):
        """This function computes the indices for the base parameters. The number of base parameters is obtained as the
        maximum obtainable rank of the observation matrix using a set of randomly generated dummy observations for the robot
        trajectory. The specific indices for the base parameters are obtained by conducting a QR decomposition of the
        observation matrix and analyzing the diagonal elements of the upper triangular (R) matrix."""
        # TODO: Change code such that 'dummy_pos, 'dummy_vel', and 'dummy_acc' can be called with a 2D list of
        #  indices to produce a 2D list of 'dummy_pos', 'dummy_vel', and 'dummy_acc'. This way, the regressor can be
        #  called with a 2D np.array(), which will drastically speed up the computation of the dummy observation matrix.

        # The set of dummy observations
        dummy_pos = np.array([-np.pi, -0.5 * np.pi, -0.25 * np.pi, 0.0, 0.25 * np.pi, 0.5 * np.pi, np.pi])
        dummy_vel = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
        dummy_acc = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
        assert len(dummy_pos) == len(dummy_vel) == len(dummy_acc)

        n_rank_evals = 0  # Loop counter for observation matrix concatenations
        rank_W = []

        random_idx_init = [[np.random.randint(self.n_joints + 1) for _ in range(self.n_joints)] for _ in range(3)]
        dummy_args_init = np.concatenate((dummy_pos[random_idx_init[0][:]], dummy_vel[random_idx_init[1][:]],
                                          dummy_acc[random_idx_init[2][:]]))

        W_N = regressor_with_instantiated_parameters(*dummy_args_init)
        W = regressor_with_instantiated_parameters(*dummy_args_init)

        # Continue the computations while the rank of the observation matrix 'W' keeps improving
        while n_rank_evals < 3 or rank_W[-1] > rank_W[-2]:  # While the rank of the observation matrix keeps increasing
            # Generate random indices for the dummy observations
            random_idx = [[[np.random.randint(self.n_joints + 1) for _ in range(self.n_joints)]
                           for _ in range(RigidBodyDynamics.__n_regressor_evals_per_rank_calculation)] for _ in range(3)]

            # Evaluate the regressor in a number of dummy observations and vertically stack the regressor matrices
            for i in range(RigidBodyDynamics.__n_regressor_evals_per_rank_calculation):
                dummy_args = np.concatenate(
                    (dummy_pos[random_idx[0][i][:]], dummy_vel[random_idx[1][i][:]], dummy_acc[random_idx[2][i][:]]))

                reg_i = regressor_with_instantiated_parameters(*dummy_args)  # Each index contains a (n_joints x n_par) regressor matrix
                W_N = np.append(W_N, reg_i, axis=0)
            W = np.append(W, W_N, axis=0)

            # Evaluate rank of observation matrix
            rank_W.append(np.linalg.matrix_rank(W))
            print(f"Rank of observation matrix: {rank_W[-1]}")

            n_rank_evals += 1
            if n_rank_evals > RigidBodyDynamics.__max_number_of_rank_evaluations:
                raise Exception(f"Numerical estimation of the number of base inertial parameters did not converge within {n_rank_evals * RigidBodyDynamics.__n_regressor_evals_per_rank_calculation} regressor evaluations.")

        r = np.linalg.qr(W, mode='r')
        idx_is_base = abs(np.diag(r)) > RigidBodyDynamics.__qr_numerical_threshold
        idx_base = np.where(idx_is_base)[0].tolist()
        assert len(idx_base) == rank_W[-1]

        return idx_base

    def evaluate_dynamics_basis(self, q_num, qd_num, qdd_num):
        """
        This method evaluates the rigid-body dynamics basis with instantiated DH parameters, gravity, and
        TCP force/torque. The rigid-body parameters are set equal to ones.
        """
        assert q_num.shape == qd_num.shape == qdd_num.shape

        rbd_reg = self.regressor()
        tauJ_basis = [sp.summation(rbd_reg[j, :]) for j in range(rbd_reg.shape[0])]
        args_sym = self.q[1:] + self.qd[1:] + self.qdd[1:]
        args_num = np.concatenate((q_num, qd_num, qdd_num))
        tauJ_num = np.zeros((len(tauJ_basis), args_num.shape[1]))
        sys.setrecursionlimit(int(1e6))  # Prevents errors in sympy lambdify

        # ******************************************* PARALLEL COMPUTATION *********************************************
        # js = list(range(self.n_joints + 1))
        # tau_per_task = [tauJ_basis[j] for j in js]  # Allows one to control how many tasks by controlling how many js
        # data_per_task = list(product(zip(tau_per_task, js), [self.__parameters_linear()]))
        #
        # with Pool() as p:
        #     tauJ_basis = p.map(compute_dynamics, data_per_task)
        # **************************************************************************************************************

        for j in range(self.n_joints + 1):
            print(f"Computing lamdified expression RobotDynamics.evaluate()[j={j}]...")
            tauJ_j_function = sp.lambdify(args_sym, tauJ_basis[j], 'numpy')
            print(f"Evaluating numerically RobotDynamics.evaluate()[j={j}]...")
            tauJ_num[j, :] = tauJ_j_function(*args_num)
        print(f"Successfully computed RobotDynamics.evaluate()...")
        return tauJ_num

    def observation_matrix_joint(self, j, q_num, qd_num, qdd_num):
        assert q_num.shape == qd_num.shape == qdd_num.shape
        assert 0 < j <= self.n_joints

        n_samples = q_num.shape[1]
        regressor_j = self.regressor_joint(j)
        sys.setrecursionlimit(int(1e6))
        nonzeros = [not elem.is_zero for elem in regressor_j]
        args_sym = self.q[1:] + self.qd[1:] + self.qdd[1:]
        observation_matrix_j = np.zeros((n_samples, regressor_j.shape[1]))
        regressor_j_nonzeros_fcn = sp.lambdify(args_sym, regressor_j[:, nonzeros], 'numpy')
        args_num = np.concatenate((q_num, qd_num, qdd_num))
        observation_matrix_j[:, nonzeros] = regressor_j_nonzeros_fcn(*args_num).squeeze().transpose()
        return observation_matrix_j

    def observation_matrix(self, q_num, qd_num, qdd_num):
        assert q_num.shape == qd_num.shape == qdd_num.shape

        n_samples = q_num.shape[1]
        args_sym = self.q[1:] + self.qd[1:] + self.qdd[1:]
        args_num = np.concatenate(q_num, qd_num, qdd_num)
        regressor = self.regressor()
        observation_matrix = np.zeros((self.n_joints * n_samples, sum(self.number_of_parameters())))
        sys.setrecursionlimit(int(1e6))
        for j in range(self.n_joints):
            nonzeros_j = [not elem.is_zero for elem in regressor[j, :]]
            regressor_j_nonzeros_fcn = sp.lambdify(args_sym, regressor[j, nonzeros_j], 'numpy')
            observation_matrix[j*n_samples:(j+1)*n_samples, nonzeros_j] = regressor_j_nonzeros_fcn(*args_num)

        return np.vstack([self.observation_matrix_joint(j, q_num, qd_num, qdd_num) for j in range(self.n_joints)])

    def regressor_joint(self, j):
        return cache_object(self.filepath_regressor_joint(j), lambda: self.regressor()[j, :])

    def regressor(self):
        def compute_regressor():
            regressor_linear_exist = self.__regressor_linear_with_instantiated_parameters(robot_parameter_function=get_ur5e_parameters)

            args_sym = self.q[1:] + self.qd[1:] + self.qdd[1:]  # list concatenation
            sys.setrecursionlimit(int(1e6))  # Prevents errors in sympy lambdify
            regressor_linear_exist_func = sp.lambdify(args_sym, regressor_linear_exist, 'numpy')

            parameter_indices_base = self.__indices_base_exist(regressor_linear_exist_func)

            for j in range(1, self.n_joints+1):
                cache_object(from_cache(f"rigid_body_dynamics_regressor_joint_{j}"),
                             lambda: regressor_linear_exist[j, parameter_indices_base])

            return regressor_linear_exist[1:, parameter_indices_base]

        return cache_object(self.filepath_regressor, compute_regressor)

    def dynamics(self):
        def compute_dynamics_and_replace_first_moments():
            rbd = compute_torques_symbolic_ur(self.q, self.qd, self.qdd, self.__f_tcp, self.__n_tcp,self.__i_cor,
                                              sp.Matrix(self.gravity), inertia_is_wrt_CoM=False)
            js = list(range(self.n_joints + 1))
            dynamics_per_task = [rbd[j] for j in js]  # Allows one to control how many tasks by controlling how many js.
            data_per_task = list(product(zip(dynamics_per_task, js), [self.__m], [self.__pc], [self.__m_pc], [self.n_joints]))

            with Pool() as p:
                rbd_lin = p.map(replace_first_moments, data_per_task)
            return rbd_lin

        return cache_object(self.filepath_dynamics, compute_dynamics_and_replace_first_moments)
