import numpy as np
import sympy as sp
from multiprocessing import Pool
from itertools import product, chain
import sys

from aurt.file_system import cache_object, from_cache
from aurt.dynamics_aux import sym_mat_to_subs, replace_first_moments, compute_regressor_row
from aurt.num_sym_layers import spvector, spcross, spdot
from aurt.data_processing import ModifiedDH


class RigidBodyDynamics:
    __qr_numerical_threshold = 1e-12  # Numerical threshold used in identifying the base inertial parameters
    __max_number_of_rank_evaluations = 100  # Max. no. of rank evaluations in computing the base inertial parameters
    __n_regressor_evals_per_rank_calculation = 2  # No. of regressor evaluations per rank calculation
    __min_rank_evals = 4

    def __init__(self, modified_dh: ModifiedDH, gravity=None, tcp_force_torque=None):
        self.mdh = modified_dh
        self.n_joints = modified_dh.n_joints

        self.q = self.mdh.q
        self.qd = [sp.Integer(0)] + [sp.symbols(f"qd{j}") for j in range(1, self.n_joints + 1)]
        self.qdd = [sp.Integer(0)] + [sp.symbols(f"qdd{j}") for j in range(1, self.n_joints + 1)]

        self.__m = sp.symbols([f"m{j}" for j in range(self.n_joints + 1)])
        self.__mX = sp.symbols([f"mX{j}" for j in range(self.n_joints + 1)])
        self.__mY = sp.symbols([f"mY{j}" for j in range(self.n_joints + 1)])
        self.__mZ = sp.symbols([f"mZ{j}" for j in range(self.n_joints + 1)])

        # TODO: clean code
        PC = [spvector([0.0, 0.0, 0.0])] * (self.n_joints + 1)
        for j in range(1, self.n_joints + 1):
            PC[j] = spvector([sp.symbols(f"PCx{j}"), sp.symbols(f"PCy{j}"), sp.symbols(f"PCz{j}")])
        self.__pc = PC

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
        #self.filepath_dynamics = from_cache('rigid_body_dynamics')
        #self.__filepath_regressor_joint = from_cache('rigid_body_dynamics_regressor_joint_')

        self.n_params = None
        self.params = None
        self.regressor_linear = None
        self.idx_base_exist = None

    def filepath_regressor_joint(self, j):
        return from_cache(f'rigid_body_dynamics_regressor_joint_{j}')

    @property
    def n_params(self):
        if self._n_params is None:
            self.number_of_parameters()
        return self._n_params

    @property
    def params(self):
        if self._params is None:
            self.parameters()
        return self._params

    @n_params.setter
    def n_params(self, val):
        self._n_params = val

    @params.setter
    def params(self, val):
        self._params = val

    def parameters(self):
        """
        Returns a list of 'n_joints + 1' elements with each element comprising a list of all rigid body parameters
        related to that corresponding link.
        """
        base_params_information = self.__base_parameters_information()
        self.params = base_params_information[2][1:]
        self.n_params = base_params_information[1][1:]
        return self.params

    def number_of_parameters(self):
        self.n_params = self.__base_parameters_information()[1][1:]
        return self.n_params

    def __parameters_linear(self):
        """
        Returns a list of 'n_joints + 1' elements with each element comprising a list of all parameters related to that
        corresponding rigid body.
        """
        return [[self.__XX[j], self.__XY[j], self.__XZ[j], self.__YY[j], self.__YZ[j], self.__ZZ[j],
                 self.__mX[j], self.__mY[j], self.__mZ[j], self.__m[j]] for j in range(self.n_joints + 1)]

    def __regressor_linear(self):
        if self.regressor_linear is not None:
            return self.regressor_linear

        dynamics_linearizable = self.dynamics()

        js = list(range(self.n_joints + 1))
        tau_per_task = [dynamics_linearizable[j] for j in js]  # Allows one to control how many tasks by controlling how many js
        data_per_task = list(product(zip(tau_per_task, js), [self.n_joints], [self.__parameters_linear()]))

        with Pool() as p:
            reg = p.map(compute_regressor_row, data_per_task)

        res = sp.Matrix(reg)
        self.regressor_linear = res
        return res

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
                lambda: self.__regressor_linear_with_instantiated_parameters()))
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

    def __numerical_alpha_to_symbolical_pi(self):
        alpha_sym = []
        for i in range(len(self.mdh.alpha)):
            pi_factor = self.mdh.alpha[i] / (np.pi / 2)
            if abs(round(pi_factor) - pi_factor) < 1e-2:
                alpha_sym.append(round(pi_factor)*sp.pi/2)
            else:
                alpha_sym.append(sp.symbols(f"alpha{i}"))
        return alpha_sym

    def __mdh_num_to_sym(self):
        d = [sp.symbols(f"d{i}") if d != 0 else sp.Integer(0) for i, d in enumerate(self.mdh.d)]
        a = [sp.symbols(f"a{i}") if a != 0 else sp.Integer(0) for i, a in enumerate(self.mdh.a)]
        alpha = self.__numerical_alpha_to_symbolical_pi()
        m = [0] * (self.n_joints + 1)
        for j in range(1, self.n_joints + 1):
            m[j] = sp.symbols(f"m{j}")
        return m, d, a, alpha

    def __regressor_linear_with_instantiated_parameters(self):
        _, d, a, _ = self.__mdh_num_to_sym()

        def load_regressor_and_subs():
            regressor_reduced = self.__regressor_linear_exist()
            return regressor_reduced.subs(
                sym_mat_to_subs([a, d, self.__g, self.__f_tcp, self.__n_tcp],
                                [self.mdh.a, self.mdh.d, self.gravity, self.__f_tcp_num, self.__n_tcp_num]))

        return load_regressor_and_subs()

    def __indices_base_exist(self, regressor_with_instantiated_parameters):
        """This function computes the indices for the base parameters. The number of base parameters is obtained as the
        maximum obtainable rank of the observation matrix using a set of randomly generated dummy observations for the robot
        trajectory. The specific indices for the base parameters are obtained by conducting a QR decomposition of the
        observation matrix and analyzing the diagonal elements of the upper triangular (R) matrix."""
        # TODO: Change code such that 'dummy_pos, 'dummy_vel', and 'dummy_acc' can be called with a 2D list of
        #  indices to produce a 2D list of 'dummy_pos', 'dummy_vel', and 'dummy_acc'. This way, the regressor can be
        #  called with a 2D np.array(), which will drastically speed up the computation of the dummy observation matrix.

        if self.idx_base_exist is not None:
            return self.idx_base_exist

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
        while n_rank_evals < RigidBodyDynamics.__min_rank_evals or rank_W[-1] > rank_W[-3]:  # While the rank of the observation matrix keeps increasing
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

        self.idx_base_exist = idx_base
        return idx_base

    # def observation_matrix_joint_parameters_for_joint(self, j, j_par, q_num, qd_num, qdd_num):
    #     assert q_num.shape == qd_num.shape == qdd_num.shape
    #     assert 0 <= j < self.n_joints
    #     assert 0 <= j_par < self.n_joints

    #     regressor_j_jpar = self.regressor_joint_parameters_for_joint(j, j_par)
    #     args_sym = self.q[1:] + self.qd[1:] + self.qdd[1:]
    #     nonzeros = [not elem.is_zero for elem in regressor_j_jpar]
    #     sys.setrecursionlimit(int(1e6))
    #     regressor_j_jpar_nonzeros_fcn = sp.lambdify(args_sym, regressor_j_jpar[:, nonzeros], 'numpy')

    #     n_samples = q_num.shape[1]
    #     observation_matrix_j = np.zeros((n_samples, regressor_j_jpar.shape[1]))
    #     args_num = np.concatenate((q_num, qd_num, qdd_num))
    #     observation_matrix_j[:, nonzeros] = regressor_j_jpar_nonzeros_fcn(*args_num).transpose().squeeze(axis=2)
    #     return observation_matrix_j

    def observation_matrix_joint(self, j, q_num, qd_num, qdd_num):
        assert q_num.shape == qd_num.shape == qdd_num.shape
        assert 0 <= j < self.n_joints

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

    # def observation_matrix(self, q_num, qd_num, qdd_num):
    #     assert q_num.shape == qd_num.shape == qdd_num.shape

    #     return np.vstack([self.observation_matrix_joint(j, q_num, qd_num, qdd_num) for j in range(self.n_joints)])

    def regressor_joint_parameters_for_joint(self, j, par_j):
        column_idx_start = sum(self.number_of_parameters()[:par_j])
        column_idx_end = column_idx_start + self.number_of_parameters()[par_j]
        res = self.regressor_joint(j)[:, column_idx_start:column_idx_end]
        return res

    def regressor_joint(self, j):
        return cache_object(self.filepath_regressor_joint(j+1), lambda: self.regressor()[j, :])

    def regressor(self, output_filename="rigid_body_dynamics_regressor"):
        filepath_regressor = from_cache(output_filename)

        def compute_regressor():
            regressor_linear_exist = self.__regressor_linear_with_instantiated_parameters()

            args_sym = self.q[1:] + self.qd[1:] + self.qdd[1:]  # list concatenation
            sys.setrecursionlimit(int(1e6))  # Prevents errors in sympy lambdify
            regressor_linear_exist_func = sp.lambdify(args_sym, regressor_linear_exist, 'numpy')

            parameter_indices_base = self.__indices_base_exist(regressor_linear_exist_func)

            for j in range(self.n_joints):
                cache_object(self.filepath_regressor_joint(j+1), lambda: regressor_linear_exist[j+1, parameter_indices_base])
            
            return regressor_linear_exist[1:, parameter_indices_base]

        return cache_object(filepath_regressor, compute_regressor)

    def dynamics(self):
        def compute_dynamics_and_replace_first_moments():
            rbd = self.__rigid_body_dynamics()
            js = list(range(self.n_joints + 1))
            dynamics_per_task = [rbd[j] for j in js]  # Allows one to control how many tasks by controlling how many js.
            data_per_task = list(product(zip(dynamics_per_task, js), [self.__m], [self.__pc], [self.__m_pc], [self.n_joints]))

            with Pool() as p:
                rbd_lin = p.map(replace_first_moments, data_per_task)
            return rbd_lin

        return cache_object(from_cache('rigid_body_dynamics'), compute_dynamics_and_replace_first_moments)

    def __get_P(self, a, d, alpha):
        """
        P[i] means the position of frame i wrt to i-1
        P[2] = [d, -0.2]^T
        P[5] = [a[5-1], 0, d[5]] means the position of frame 5 wrt to 4
        """
        P = [spvector([0, 0, 0]) for i in range(0, self.n_joints + 2)]

        for i in range(1, self.n_joints + 1):
            P[i] = spvector([a[i - 1], -sp.sin(alpha[i - 1]) * d[i], sp.cos(alpha[i - 1]) * d[i]])

        return P

    def __get_forward_kinematics(self, alpha):
        c = lambda i: sp.cos(self.q[i])
        s = lambda i: sp.sin(self.q[i])

        R_i_im1 = [sp.zeros(3, 3) for i in range(self.n_joints + 2)]
        for i in range(1, self.n_joints + 1):
            # i=1,...,6
            R_i_im1[i] = sp.Matrix([
                [c(i),                    -s(i),                   0],
                [s(i) * sp.cos(alpha[i-1]),  c(i) * sp.cos(alpha[i-1]),  -sp.sin(alpha[i-1])],
                [s(i) * sp.sin(alpha[i-1]),  c(i) * sp.sin(alpha[i-1]),  sp.cos(alpha[i-1])]
            ])

        # R_i_im1[1] means the orientation of frame 1 wrt to 0.
        # R_i_im1[4] means the orientation of frame 4 wrt to 3.
        # R_i_im1[4+1] = orientation of frame 5 to frame 4

        R_im1_i = [r.transpose() for r in R_i_im1]
        # R_im1_i[4] means the rotation from frame 3 to 4.

        return R_i_im1, R_im1_i

    def __rigid_body_dynamics(self):
        """
        Follows algorithm described in:
        Craig, John J. 2009. "Introduction to Robotics: Mechanics and Control", 3/E. Pearson Education India.
        """

        identity_3 = sp.eye(3)
        (m, d, a, alpha) = self.__mdh_num_to_sym()
        P = self.__get_P(a, d, alpha)  # position vectors
        PC = self.__pc  # center-of-mass locations

        I_CoM = [sp.zeros(3, 3) for _ in range(self.n_joints + 1)]
        for j in range(1, self.n_joints + 1):
            PC_dot_left = spdot(PC[j].transpose(), PC[j])
            PC_dot_right = spdot(PC[j], (PC[j].transpose()))
            assert PC_dot_left.shape == (1, 1)
            assert PC_dot_right.shape == (3, 3)
            PC_dot_scalar = PC_dot_left[0, 0]
            I_CoM[j] = self.__i_cor[j] - m[j] * (PC_dot_scalar * identity_3 - PC_dot_right)

        # State
        w = [spvector([0, 0, 0]) for _ in range(self.n_joints + 1)]  # Angular velocity
        wd = [spvector([0, 0, 0]) for _ in range(self.n_joints + 1)]  # Angular acceleration
        vd = [spvector([0, 0, 0]) for _ in range(self.n_joints + 1)]  # Translational acceleration

        # Gravity
        vd[0] = -self.__g

        vcd = [spvector([0, 0, 0]) for _ in range(self.n_joints + 1)]
        F = [spvector([0, 0, 0]) for _ in range(self.n_joints + 1)]
        N = [spvector([0, 0, 0]) for _ in range(self.n_joints + 1)]

        f = [spvector([0, 0, 0]) for _ in range(self.n_joints + 1)]
        f.append(self.__f_tcp)

        n = [spvector([0, 0, 0]) for _ in range(self.n_joints + 1)]
        n.append(self.__n_tcp)

        Z = spvector([0, 0, 1])
        (R_i_im1, R_im1_i) = self.__get_forward_kinematics(alpha)

        # Outputs
        tau = [sp.zeros(1, 1) for _ in range(self.n_joints + 1)]

        # Outward recursion j: 0 -> 5
        for j in range(self.n_joints):
            w[j+1] = spdot(R_im1_i[j+1], w[j]) + self.qd[j+1] * Z
            assert w[j+1].shape == (3, 1)
            wd[j+1] = spdot(R_im1_i[j+1], wd[j]) + spcross(spdot(R_im1_i[j+1], w[j]), self.qd[j+1] * Z) + self.qdd[j+1] * Z
            assert wd[j+1].shape == (3, 1)
            vd[j+1] = spdot(R_im1_i[j+1], spcross(wd[j], P[j+1]) + spcross(w[j], spcross(w[j], P[j+1])) + vd[j])
            assert vd[j+1].shape == (3, 1)
            vcd[j+1] = spcross(wd[j+1], PC[j+1]) + spcross(w[j+1], spcross(w[j+1], PC[j+1])) + vd[j+1]
            assert vcd[j+1].shape == (3, 1)
            F[j+1] = m[j+1] * vcd[j+1]
            assert F[j+1].shape == (3, 1)
            N[j+1] = spdot(I_CoM[j+1], wd[j+1]) + spcross(w[j+1], spdot(I_CoM[j+1], w[j+1]))
            assert N[j+1].shape == (3, 1)

        # Inward recursion j: 6 -> 1
        for j in reversed(range(1, self.n_joints + 1)):
            f[j] = spdot(R_i_im1[j+1], f[j+1]) + F[j]
            n[j] = N[j] + spdot(R_i_im1[j+1], n[j+1]) + spcross(PC[j], F[j]) + spcross(P[j+1], spdot(R_i_im1[j+1], f[j+1]))
            assert n[j].shape == (3, 1), n[j]
            tau[j] = spdot(n[j].transpose(), Z)
            assert tau[j].shape == (1, 1)

        tau_list = [t[0, 0] for t in tau]

        return tau_list


    def plot_kinematics(self, block=True):
        try:
            import roboticstoolbox as rtb
            MDHRobot = self._create_kinematics_mdh(rtb)
            robot_kinematics = MDHRobot(self.mdh)
            robot_kinematics.plot(robot_kinematics.q, block=block)
        except ImportError:
            import warnings
            warnings.warn("The roboticstoolbox package is not installed, please install it for plotting the robot kinematics.")

    
    def _create_kinematics_mdh(self, rtb):
        class MDHRobot(rtb.DHRobot):
            def __init__(self, mdh: ModifiedDH):
                links = []
                n_links = []
                for i in range(mdh.n_joints):
                    d = mdh.d[i]
                    a = mdh.a[i]
                    alpha = mdh.alpha[i]
                    setattr(self, f"link_{i}", rtb.RevoluteMDH(d=d, a=a, alpha=alpha))
                    links.append(getattr(self, f"link_{i}"))
                    n_links.append(i+1)
                super().__init__(links, name="robot")
        return MDHRobot


