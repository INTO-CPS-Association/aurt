import sympy as sp


class SymbolicModel:
    # TODO: Make classes 'RigidBodyDynamicsModel' and 'JointDynamicsModel'
    def __init__(self, modified_dh, gravity_direction=None, viscous_friction_powers=None, friction_load_model=None):
        self.__mdh = modified_dh
        if gravity_direction is None:
            print(f"No gravity direction was specified. Assuming the first joint axis of rotation to be parallel to gravity...")
            self.__gravity_direction = [0, 0, -1]  # change recursive Newton-Euler to fit this convention
        self.friction_load_model = 'none'  # 'none', 'square' or 'abs'
        self.friction_hysteresis_model = 'sgn'  # 'sgn' or 'gms'

        # Viscous friction
        if viscous_friction_powers is None:
            self.viscous_friction_powers = [1]
        else:
            self.viscous_friction_powers = viscous_friction_powers  # [1, 2, 3, ...], i.e. list of positive integers

        # Load-dependent friction model
        self.friction_load_model = friction_load_model  # 'none', 'square', or 'abs'

        self.n_joints = SymbolicModel.number_of_joints(modified_dh)

    def __rigid_body_dynamics_parameters(self):
        """
        Returns a list of 'n_joints + 1' elements with each element comprising a list of all rigid body parameters related to
        that corresponding link.
        """
        m = sp.symbols([f"m{j}" for j in range(self.n_joints + 1)])
        mX = sp.symbols([f"mX{j}" for j in range(self.n_joints + 1)])
        mY = sp.symbols([f"mY{j}" for j in range(self.n_joints + 1)])
        mZ = sp.symbols([f"mZ{j}" for j in range(self.n_joints + 1)])
        XX = sp.symbols([f"XX{j}" for j in range(self.n_joints + 1)])
        XY = sp.symbols([f"XY{j}" for j in range(self.n_joints + 1)])
        XZ = sp.symbols([f"XZ{j}" for j in range(self.n_joints + 1)])
        YY = sp.symbols([f"YY{j}" for j in range(self.n_joints + 1)])
        YZ = sp.symbols([f"YZ{j}" for j in range(self.n_joints + 1)])
        ZZ = sp.symbols([f"ZZ{j}" for j in range(self.n_joints + 1)])

        return [[XX[j], XY[j], XZ[j], YY[j], YZ[j], ZZ[j], mX[j], mY[j], mZ[j], m[j]] for j in range(self.n_joints + 1)]

    def regressor(self):
        return 1

    def __replace_first_moments(self, args):

        tau_sym_j = args[0][0]  # tau_sym[j]
        j = args[0][1]
        m = args[1]  # [m1, ..., mN]
        PC = args[2]  # PC = [PC[1], ..., PC[N]], PC[j] = [PCxj, PCyj, PCzj]
        m_pc = args[3]  # mPC = [mPC[1], ..., mPC[N]], mPC[j] = [mXj, mYj, mZj]

        # TODO: REMOVE THE 'IGNORE J=0' STUFF
        if j == 0:
            print(f"Ignore joint {j}...")
            return tau_sym_j

        n_cartesian = len(m_pc[0])
        for jj in range(j, self.n_joints+1):  # The joint j torque equation is affected by dynamic coefficients only for links jj >= j
            for i in range(n_cartesian):  # Cartesian coordinates loop
                # Print progression counter
                total = (self.n_joints+1-j)*n_cartesian
                progress = (jj - j)*n_cartesian + i + 1
                print(f"Task {j}: {progress}/{total} (tau[{j}]: {m[jj]}*{PC[jj][i]} -> {m_pc[jj][i]})")
                tau_sym_j = tau_sym_j.expand().subs(m[jj] * PC[jj][i], m_pc[jj][i])  # expand() is - for unknown reasons - needed for subs() to consistently detect the "m*PC" products

        return tau_sym_j

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
        reg_j = sp.zeros(1, sum(n_par_linear))  # Initialization

        if j == 0:
            print(f"Ignore joint {j}...")
            return reg_j

        # For joint j, we loop through the parameters belonging to joints/links >= j. This is because it is physically
        # impossible for torque eq. j to include a parameter related to proximal links (< j). We divide the parameter
        # loop (for joints >= j) in two variables 'jj' and 'i':
        #   'jj' describes the joint, which the parameter belongs to
        #   'i'  describes the parameter's index/number for joint jj.
        for jj in range(j, self.n_joints + 1):  # Joint loop including this and distal (succeeding) joints (jj >= j)
            for i in range(n_par_linear[jj]):  # joint jj parameter loop
                column_idx = sum(n_par_linear[:jj]) + i
                print(f"Computing regressor(row={j}/{self.n_joints + 1}, column={column_idx}/{sum(n_par_linear)}) by analyzing dependency of tau[{j}] on joint {jj}'s parameter {i}: {p_linear[jj][i]}")
                reg_j[0, column_idx] = sp.diff(tau_sym_linearizable_j, p_linear[jj][i])

        return reg_j

    @staticmethod
    def number_of_joints(mdh):
        return mdh.shape[0]
