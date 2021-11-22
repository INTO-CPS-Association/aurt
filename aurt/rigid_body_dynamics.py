import numpy as np
import sympy as sp
from multiprocessing import Pool
from itertools import product, chain
from inspect import signature
import sys
from logging import Logger

from aurt.linear_system import LinearSystem
from aurt.caching import Cache
from aurt.dynamics_aux import list_2D_to_sympy_vector, sym_mat_to_subs
from aurt.num_sym_layers import spvector, spcross, spdot
from aurt.data_processing import ModifiedDH


class RigidBodyDynamics(LinearSystem):
    _qr_numerical_threshold = 1e-12  # Numerical threshold used in identifying the Base Inertial Parameters (BIP)
    _max_number_of_rank_evaluations = 100  # Max. no. of rank evaluations in computing the BIP
    _n_rank_convergence = 3  # No. of rank calculations in which the rank should not increase to assume convergence of the BIP identification
    _n_regressor_evals_per_rank_calculation = 1  # No. of regressor evaluations per rank calculation
    _min_rank_evals = 4
    _filename_regressor_joint = "rigid_body_dynamics_regressor_joint"

    def __init__(self, logger: Logger, modified_dh: ModifiedDH, cache: Cache, multi_processing: bool=True):
        super().__init__(logger, cache, name="rigid-body dynamics")

        self._g = [*sp.symbols("gx gy gz")]
        self._g_num = None

        self.multi_processing = multi_processing
        self.mdh = modified_dh
        self.n_joints = modified_dh.n_joints

        self.q = self.mdh.q
        self.qd = [sp.Integer(0)] + [sp.symbols(f"qd{j}") for j in range(1, self.n_joints + 1)]
        self.qdd = [sp.Integer(0)] + [sp.symbols(f"qdd{j}") for j in range(1, self.n_joints + 1)]

        self._m = sp.symbols(f"m:{self.n_joints + 1}")
        self._mX = sp.symbols(f"mX:{self.n_joints + 1}")
        self._mY = sp.symbols(f"mY:{self.n_joints + 1}")
        self._mZ = sp.symbols(f"mZ:{self.n_joints + 1}")

        self._XX = sp.symbols(f"XX:{self.n_joints + 1}")
        self._XY = sp.symbols(f"XY:{self.n_joints + 1}")
        self._XZ = sp.symbols(f"XZ:{self.n_joints + 1}")
        self._YY = sp.symbols(f"YY:{self.n_joints + 1}")
        self._YZ = sp.symbols(f"YZ:{self.n_joints + 1}")
        self._ZZ = sp.symbols(f"ZZ:{self.n_joints + 1}")
        self._i_cor = [sp.zeros(3, 3) for _ in range(self.n_joints + 1)]
        for j in range(self.n_joints + 1):
            self._i_cor[j] = sp.Matrix([
                [self._XX[j], self._XY[j], self._XZ[j]],
                [self._XY[j], self._YY[j], self._YZ[j]],
                [self._XZ[j], self._YZ[j], self._ZZ[j]]
            ])
        
        PC = [spvector([0, 0, 0]) for _ in range(self.n_joints + 1)]
        for j in range(1, self.n_joints + 1):
            PC[j] = spvector([sp.symbols(f"PCx{j}"), sp.symbols(f"PCy{j}"), sp.symbols(f"PCz{j}")])
        self._pc = PC
        self._m_pc = [[self._mX[j], self._mY[j], self._mZ[j]] for j in range(self.n_joints + 1)]

        # fx, fy, fz, nx, ny, nz = sp.symbols(f"fx fy fz nx ny nz")
        self._f_tcp = sp.zeros(3, 1)  # sp.Matrix([fx, fy, fz])  # Force at the TCP
        self._n_tcp = sp.zeros(3, 1)  # sp.Matrix([nx, ny, nz])  # Moment at the TCP

        # if tcp_force_torque is None:
        #     f_tcp_num = np.array([0, 0, 0])
        #     n_tcp_num = np.array([0, 0, 0])
        #     tcp_force_torque = [f_tcp_num, n_tcp_num]
        # self._f_tcp_num = tcp_force_torque[0]
        # self._n_tcp_num = tcp_force_torque[1]

        super().compute_linearly_independent_system()

    @staticmethod
    def filepath_regressor_joint(j):
        return f"{RigidBodyDynamics._filename_regressor_joint}_{j}"

    def states(self):
        if self._g_num is None:
            return list(map(list, zip(self.q[1:], self.qd[1:], self.qdd[1:]))) + [self._g]
        else:
            return list(map(list, zip(self.q[1:], self.qd[1:], self.qdd[1:])))

    def _parameters_full(self):
        """
        A list of 'n_joints + 1' elements with each element comprising a list of all Standard Inertial 
        Parameters (SIP) related to that corresponding rigid body.
        """
        return [[self._XX[j], self._XY[j], self._XZ[j], self._YY[j], self._YZ[j], self._ZZ[j],
                     self._mX[j], self._mY[j], self._mZ[j], self._m[j]] for j in range(1, self.n_joints + 1)]

    def _regressor_sip_full(self):
        """
        Constructs the regressor corresponding to the full set of Standard Inertial Parameters (SIP) based 
        on the dynamics formulated in terms of the SIP. 
        """
        dynamics_sip = self.dynamics_sip()

        js = list(range(self.n_joints))
        tau_per_task = [dynamics_sip[j] for j in js]  # Allows one to control how many tasks by controlling how many js
        data_per_task = list(product(zip(tau_per_task, js), [self.n_joints], [self._parameters_full()]))
        
        if self.multi_processing:
            with Pool() as p:
                reg = p.map(self._compute_regressor_row, data_per_task)
        else:
            p_sip_full = self._parameters_full()
            n_par = [len(p_sip_full[j]) for j in range(len(p_sip_full))]
            reg = sp.zeros(self.n_joints, sum(n_par))
            for j in range(self.n_joints):
                reg[j, :] = self._compute_regressor_row(data_per_task[j])

        res = sp.Matrix(reg)
        return res
    
    def _replace_first_moments(self, args):
        """
        Replaces the products of masses and center-of-mass locations with equivalent parameters 
        related to the first moments of mass, i.e.\n
            m[j] * Pc[j, i] -> mPc[j, i],     j = 1, ..., n_joints,  i = 1, 2, 3.
        """
        tau_sym_j = args[0][0]  # tau_sym[j]
        j = args[0][1]
        m = args[1]  # [m1, ..., mN]
        pc = args[2]  # PC = [PC[1], ..., PC[N]], PC[j] = [PCxj, PCyj, PCzj]
        m_pc = args[3]  # mPC = [mPC[1], ..., mPC[N]], mPC[j] = [mXj, mYj, mZj]
        n_joints = args[4]

        self.logger.info("Replacing products of masses and center-of-mass locations with first moments of mass...")

        # TODO: REMOVE THE 'IGNORE J=0' STUFF
        if j == 0:
            self.logger.info(f"Ignore joint {j}...")
            return tau_sym_j
        
        n_cartesian = len(m_pc[0])
        for jj in range(j, n_joints + 1):  # The joint j torque equation is affected by dynamic coefficients only for links jj >= j
            for i in range(n_cartesian):  # Cartesian coordinates loop
                # Print progression counter
                total = (n_joints + 1 - j) * n_cartesian
                progress = (jj - j) * n_cartesian + i + 1
                self.logger.info(f"Task {j}: {progress}/{total} (tau[{j}]: {m[jj]}*{pc[jj][i]} -> {m_pc[jj][i]})")

                # According to tests on a 6 DoF robot, having 'expand()' inside the nested for-loops is not slower than having 'expand()' outside the nested for-loops
                tau_sym_j = tau_sym_j.expand().subs(m[jj] * pc[jj][i], m_pc[jj][i])  # expand() is - for unknown reasons - needed for subs() to consistently detect the "m*PC" products

        return tau_sym_j

    def _compute_regressor_row(self, args):
        """
        We compute the regressor by extracting the coefficient to the parameter. Each torque equation must be linearizable 
        with respect to the dynamic coefficients.

        Example:
            Given the equation 'tau = sin(q)*a', tau is linear with respect to 'a' and the regressor 'sin(q)' can be
            obtained as the coefficient of 'a' in 'tau'.
        """
        tau_sym_linearizable_j = args[0][0]
        j = args[0][1]
        n_joints = args[1]
        p_linear = args[2]

        n_par = [len(p_linear[j]) for j in range(len(p_linear))]
        reg_row_j = sp.zeros(1, sum(n_par))  # Initialization

        # For joint j, we loop through the parameters belonging to joints/links >= j. This is because it is physically
        # impossible for torque eq. j to include a parameter related to proximal links (< j). We divide the parameter
        # loop (for joints >= j) in two variables 'jj' and 'i':
        #   'jj' describes the joint, which the parameter belongs to
        #   'i'  describes the parameter's index/number for joint jj.
        for jj in range(j, n_joints):  # Joint loop including this and distal (succeeding) joints (jj >= j)
            for i in range(n_par[jj]):  # joint jj parameter loop
                column_idx = sum(n_par[:jj]) + i
                self.logger.info(f"Computing regressor(row={j+1}/{n_joints}, column={column_idx - n_par[0] + 1}/{sum(n_par[1:])}) by analyzing dependency of tau[{j}] on joint {jj}'s parameter {i+1}: {p_linear[jj][i]}")
                if p_linear[jj][i] is not sp.Integer(0):
                    reg_row_j[0, column_idx] = tau_sym_linearizable_j.diff(p_linear[jj][i])
        return reg_row_j

    def _regressor_sip_instantiated_dh(self):
        d, a, _ = self._mdh_num_to_sym()

        filename = f"{self.name}_regressor_full"
        return self._cache.get_or_cache(filename, lambda: self._regressor_sip_full().subs(sym_mat_to_subs([a, d], [self.mdh.a, self.mdh.d])))

    def convert_to_payload_dynamics(self):
        """
        Converts the rigid-body dynamics to payload dynamics only. 
        """
        
        self._pc = [sp.zeros(3,1) for _ in range(self.n_joints)] + [sp.Matrix([sp.symbols("PCx PCy PCz")]).T]

        self._m = [sp.Integer(0) for _ in range(self.n_joints)] + [sp.Symbol("m")]
        self._mX = [sp.Integer(0) for _ in range(self.n_joints)] + [sp.Symbol("mX")]
        self._mY = [sp.Integer(0) for _ in range(self.n_joints)] + [sp.Symbol("mY")]
        self._mZ = [sp.Integer(0) for _ in range(self.n_joints)] + [sp.Symbol("mZ")]
        self._m_pc = [[self._mX[j], self._mY[j], self._mZ[j]] for j in range(self.n_joints + 1)]

        self._XX = [sp.Integer(0) for _ in range(self.n_joints)] + [sp.Symbol("XX")]
        self._XY = [sp.Integer(0) for _ in range(self.n_joints)] + [sp.Symbol("XY")]
        self._XZ = [sp.Integer(0) for _ in range(self.n_joints)] + [sp.Symbol("XZ")]
        self._YY = [sp.Integer(0) for _ in range(self.n_joints)] + [sp.Symbol("YY")]
        self._YZ = [sp.Integer(0) for _ in range(self.n_joints)] + [sp.Symbol("YZ")]
        self._ZZ = [sp.Integer(0) for _ in range(self.n_joints)] + [sp.Symbol("ZZ")]
        self._i_cor = [sp.zeros(3) for _ in range(self.n_joints + 1)]
        for j in range(self.n_joints + 1):
            self._i_cor[j] = sp.Matrix([
                [self._XX[j], self._XY[j], self._XZ[j]],
                [self._XY[j], self._YY[j], self._YZ[j]],
                [self._XZ[j], self._YZ[j], self._ZZ[j]]
            ])
        
        self.compute_linearly_independent_system()

    def instantiate_gravity(self, gravity):
        if not all(gravity == self._g_num):
            self._g_num = gravity
            self.compute_linearly_independent_system()
    
    def _regressor_joint_parameters_for_joint(self, j, par_j):
        """
        Computes the entire rigid-body dynamics regressor and returns those elements of 
        row 'j' related to the parameters for joint 'par_j'.
        """

        regressor = self._regressor()
        num_par = self._number_of_parameters_full()
        col_start = sum(num_par[:par_j])
        col_end = col_start + num_par[par_j]
            
        return regressor[j, col_start:col_end]

    def _regressor(self):
        if not self._g_num is None:
            regressor_filepath = f"{self.name}_gravity={self._g_num}_regressor_full"
            return self._cache.get_or_cache(regressor_filepath, lambda: self._regressor_sip_instantiated_dh().subs(sym_mat_to_subs([self._g], [self._g_num])))
        else:
            return self._regressor_sip_instantiated_dh()

    def dynamics_sip(self):
        """
        The rigid-body dynamics formulated in terms of the Standard Inertial Parameters (SIP) with\n
            SIP = [SIP_1, ..., SIP_N],\n
            SIP_j = [XXj, XYj, XZj, YYj, YZj, ZZj, mXj, mYj, mZj, mj].
        """
        def compute_dynamics_and_replace_first_moments():
            rbd = self._rigid_body_dynamics()
            js = list(range(self.n_joints + 1))
            dynamics_per_task = [rbd[j] for j in js]  # Allows one to control how many tasks by controlling how many js.
            data_per_task = list(product(zip(dynamics_per_task, js), [self._m], [self._pc], [self._m_pc], [self.n_joints]))

            if self.multi_processing:
                with Pool() as p:  # Compute using multiple processes
                    rbd_lin = p.map(self._replace_first_moments, data_per_task)
            else:
                rbd_lin = [0]*(self.n_joints + 1)
                for j in range(self.n_joints + 1):
                    rbd_lin[j] = self._replace_first_moments(data_per_task[j])
            return rbd_lin
        
        return compute_dynamics_and_replace_first_moments()[1:]

    def _position_vectors(self, a, d, alpha):
        """The position vectors defined in (3.6), page 75 in:
        Craig, John J. 2009. "Introduction to Robotics: Mechanics and Control", 3/E. Pearson Education India."""
        p = [spvector([0, 0, 0]) for _ in range(self.n_joints + 2)]
        
        for i in range(1, self.n_joints + 1):
            p[i] = spvector([a[i-1], -sp.sin(alpha[i-1]) * d[i], sp.cos(alpha[i-1]) * d[i]])

        return p

    def _rotation_matrices(self, alpha):
        c = lambda x: sp.cos(x)
        s = lambda x: sp.sin(x)
        q = self.q

        R_i_im1 = [sp.eye(3) for _ in range(self.n_joints + 2)]
        for i in range(1, self.n_joints + 1):
            R_i_im1[i] = sp.Matrix([
                [c(q[i]),                 -s(q[i]),                  0            ],
                [s(q[i]) * c(alpha[i-1]),  c(q[i]) * c(alpha[i-1]), -s(alpha[i-1])],
                [s(q[i]) * s(alpha[i-1]),  c(q[i]) * s(alpha[i-1]),  c(alpha[i-1])]
            ])

        R_im1_i = [r.transpose() for r in R_i_im1]

        return R_i_im1, R_im1_i

    def _rigid_body_dynamics(self):
        """
        Follows algorithm described in:
        Craig, John J. 2009. "Introduction to Robotics: Mechanics and Control", 3/E. Pearson Education India.
        """

        identity_3 = sp.eye(3)
        (d, a, alpha) = self._mdh_num_to_sym()
        P = self._position_vectors(a, d, alpha)
        PC = self._pc  # center-of-mass locations

        I_CoM = [sp.zeros(3) for _ in range(self.n_joints + 1)]
        for j in range(1, self.n_joints + 1):
            PC_dot_left = spdot(PC[j].transpose(), PC[j])
            PC_dot_right = spdot(PC[j], (PC[j].transpose()))
            assert PC_dot_left.shape == (1, 1)
            assert PC_dot_right.shape == (3, 3)
            PC_dot_scalar = PC_dot_left[0, 0]
            I_CoM[j] = self._i_cor[j] - self._m[j] * (PC_dot_scalar * identity_3 - PC_dot_right)

        # State
        w = [spvector([0, 0, 0]) for _ in range(self.n_joints + 1)]  # Angular velocity
        wd = [spvector([0, 0, 0]) for _ in range(self.n_joints + 1)]  # Angular acceleration
        vd = [spvector([0, 0, 0]) for _ in range(self.n_joints + 1)]  # Translational acceleration

        # Gravity
        vd[0] = -spvector(self._g)

        vcd = [spvector([0, 0, 0]) for _ in range(self.n_joints + 1)]
        F = [spvector([0, 0, 0]) for _ in range(self.n_joints + 1)]
        N = [spvector([0, 0, 0]) for _ in range(self.n_joints + 1)]

        f = [spvector([0, 0, 0]) for _ in range(self.n_joints + 1)]
        f.append(self._f_tcp)

        n = [spvector([0, 0, 0]) for _ in range(self.n_joints + 1)]
        n.append(self._n_tcp)

        Z = spvector([0, 0, 1])
        (R_i_im1, R_im1_i) = self._rotation_matrices(alpha)

        # Outputs
        tau = [sp.zeros(1, 1) for _ in range(self.n_joints + 1)]

        # Outward recursion j: 0 -> 5
        for j in range(self.n_joints):
            w[j+1] = spdot(R_im1_i[j+1], w[j]) + self.qd[j+1] * Z
            assert w[j+1].shape == (3, 1)
            wd[j+1] = spdot(R_im1_i[j+1], wd[j]) + spcross(spdot(R_im1_i[j+1], w[j]), self.qd[j+1] * Z) + self.qdd[j+1] * Z
            assert wd[j+1].shape == (3, 1)
            vd[j+1] = R_im1_i[j+1] * (spcross(wd[j], P[j+1]) + spcross(w[j], spcross(w[j], P[j+1])) + vd[j])
            assert vd[j+1].shape == (3, 1)
            vcd[j+1] = spcross(wd[j+1], PC[j+1]) + spcross(w[j+1], spcross(w[j+1], PC[j+1])) + vd[j+1]
            assert vcd[j+1].shape == (3, 1)
            F[j+1] = self._m[j+1] * vcd[j+1]
            assert F[j+1].shape == (3, 1)
            N[j+1] = spdot(I_CoM[j+1], wd[j+1]) + spcross(w[j+1], spdot(I_CoM[j+1], w[j+1]))
            assert N[j+1].shape == (3, 1)
            # print(f"R[{j+1}]:   {R_im1_i[j+1]}")
            # print(f"w[{j+1}]:   {w[j+1]}")
            # print(f"wd[{j+1}]:  {wd[j+1]}")
            # print(f"vd[{j+1}]:  {vd[j+1]}")
            # print(f"vcd[{j+1}]: {vcd[j+1]}")
            # print(f"F[{j+1}]:   {F[j+1]}")
            # print(f"N[{j+1}]:   {N[j+1]}")

        # Inward recursion j: 6 -> 1
        for j in reversed(range(1, self.n_joints + 1)):
            f[j] = spdot(R_i_im1[j+1], f[j+1]) + F[j]
            n[j] = N[j] + spdot(R_i_im1[j+1], n[j+1]) + spcross(PC[j], F[j]) + spcross(P[j+1], spdot(R_i_im1[j+1], f[j+1]))
            assert n[j].shape == (3, 1), n[j]
            tau[j] = spdot(n[j].transpose(), Z)
            assert tau[j].shape == (1, 1)
            # print(f"R[{j+1}].T:   {R_i_im1[j+1]}")
            # print(f"f[{j}]:   {f[j]}")
            # print(f"n[{j}]:   {n[j]}")
            # print(f"tau[{j}]: {tau[j]}")

        tau_list = [t[0, 0] for t in tau]

        return tau_list
    
    def euler_lagrange(self, gravity=None):
        """
        The matrices of the Euler Lagrange system describing the rigid-body dynamics.
        """
        M = self.inertia_matrix()
        C = self.coriolis_centripetal_matrix(M)
        g = self.gravity_vector(gravity)
        return M, C, g

    def inertia_matrix(self):
        """The inertia matrix formulated in terms of the Base Inertial Parameters (BIP)."""

        N = self.n_joints
        zero_vector = lambda i: sp.zeros(i, 1) # zero vector
        qd = self.qd[1:]
        rbd_bip_inertial = self.dynamics().subs(sym_mat_to_subs([qd, self._g], [zero_vector(N), zero_vector(3)]))
        inertia_matrix = sp.zeros(N)
        for j in range(N):
            for i in range(N):
                inertia_matrix[j, i] = rbd_bip_inertial[j].expand().coeff(self.qdd[i+1])
        return inertia_matrix
    
    def gravity_vector(self, g=None):
        """
        The gravity vector formulated in terms of the Base Inertial Parameters (BIP).
        If gravity g = [gx, gy, gz] is provided - either as symbolic or numerical 
        quantities - it will be instantiated.
        In the computations, we make use of the fact that the gravity vector is
        linear in the BIP.
        """

        N = self.n_joints
        reg_bip = self.regressor()
        zero_vector = lambda i: sp.zeros(i, 1)

        reg_bip_gravity = reg_bip.subs(sym_mat_to_subs([self.qd[1:], self.qdd[1:]], [zero_vector(N), zero_vector(N)]))
        if g is not None:
            reg_bip_gravity = reg_bip.subs(sym_mat_to_subs(self._g, g))

        return reg_bip_gravity * list_2D_to_sympy_vector(self.parameters)

    def coriolis_centripetal_matrix(self, M=None):
        """
        The matrix of Coriolis and centripetal terms formulated in terms of the Base Inertial Parameters 
        (BIP). This matrix is not unique. We choose here to use the Christoffel symbols of the first kind of the 
        inertia matrix 'M'. Using the Christoffel symbols preserves the skew-symmetry property of matrix 
        {d/dt(M) - 2*C}, an essential property for various control algorithms.

        For reference, see e.g.:\n
        [1] "Springer Handbook of Robotics (2016)", section 3.3.2, equation (3.43)-(3.44).
        """
        N = self.n_joints
        if M is None: # save computations
            M = self.inertia_matrix()
        q = self.q[1:]
        qd = self.qd[1:]

        C = sp.zeros(N)  # Matrix of Coriolis and centripetal terms
        c = [[[None for _ in range(N)] for _ in range(N)] for _ in range(N)]  # Christoffel symbols

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    if c[i][k][j] is None:  # For any i, c[i][j][k] == c[i][k][j]
                        c[i][j][k] = (sp.diff(M[i,j], q[k]) + sp.diff(M[i,k], q[j]) - sp.diff(M[j,k], q[i]))/sp.Integer(2)
                        C[i,j] += c[i][j][k]*qd[k]
                    else:
                        C[i,j] += c[i][k][j]*qd[k]

        return C

    def _numerical_alpha_to_symbolical_pi(self):
        alpha_sym = []
        for i in range(len(self.mdh.alpha)):
            pi_factor = self.mdh.alpha[i] / (np.pi / 2)
            if abs(round(pi_factor) - pi_factor) < 1e-2:
                alpha_sym.append(round(pi_factor)*sp.pi/2)
            else:
                alpha_sym.append(sp.symbols(f"alpha{i}"))
        return alpha_sym

    def _mdh_num_to_sym(self):
        d = [sp.symbols(f"d{i}") if d != 0 else sp.Integer(0) for i, d in enumerate(self.mdh.d)]
        a = [sp.symbols(f"a{i}") if a != 0 else sp.Integer(0) for i, a in enumerate(self.mdh.a)]
        alpha = self._numerical_alpha_to_symbolical_pi()
        return d, a, alpha

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
