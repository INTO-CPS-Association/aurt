import sympy as sp


class JointDynamics:
    def __init__(self, n_joints, load_model=None, hysteresis_model=None, viscous_powers=None):
        self.n_joints = n_joints
        self.qd = [sp.Integer(0)] + [sp.symbols(f"qd{j}") for j in range(1, self.n_joints + 1)]
        self.tauJ = sp.symbols([f"tauJ{j}" for j in range(self.n_joints + 1)])

        # Defaults
        if load_model is None:
            load_model = 'square'  # 'none', 'square' or 'abs'
        if hysteresis_model is None:
            hysteresis_model = 'sign'  # 'sign' or 'gms'
        if viscous_powers is None:
            viscous_powers = [1]  # [1, 2, 3, ...], i.e. list of positive integers

        self.load_model = load_model
        self.hysteresis_model = hysteresis_model
        self.viscous_friction_powers = viscous_powers

    def __coulomb_friction_parameters(self):
        """
        Returns a list of 'n_joints + 1' elements, each element comprising a list of Coulomb friction parameters for
        each joint.
        """

        assert self.load_model.lower() in ('none', 'square', 'abs')

        fc = sp.symbols([f"Fc{j}" for j in range(self.n_joints + 1)])
        fcl = sp.symbols([f"Fcl{j}" for j in range(self.n_joints + 1)])

        if self.load_model.lower() == 'none':
            return [[fc[j]] for j in range(self.n_joints + 1)]
        else:
            return [[fc[j], fcl[j]] for j in range(self.n_joints + 1)]

    def __viscous_friction_parameters(self):
        """
        Returns a list of 'Njoints + 1' elements, each element comprising a list of viscous friction parameters for each
        joint.
        """

        return [[sp.symbols(f"Fv{j}_{i}") for i in self.viscous_friction_powers] for j in range(self.n_joints + 1)]

    def parameters(self):
        """
        Returns a list of 'Njoints + 1' elements with each element comprising a list of joint dynamics parameters for that
        corresponding joint.
        """

        fcs = self.__coulomb_friction_parameters()
        fvs = self.__viscous_friction_parameters()
        return [[*fcs[j], *fvs[j]] for j in range(self.n_joints + 1)]

    def number_of_parameters_each_joint(self):
        return len(self.parameters()[0])

    def __coulomb_friction_basis(self):
        """
        Returns the basis (regressor) of the Coulomb friction model.
        """

        assert self.load_model.lower() in ('none', 'square', 'abs')

        h = self.__generalized_hysteresis_model()
        if self.load_model.lower() == 'none':
            return [[h[j]] for j in range(self.n_joints + 1)]
        elif self.load_model.lower() == 'square':
            return [[h[j], h[j] * self.tauJ[j] ** 2] for j in range(self.n_joints + 1)]
        else:  # load_model == 'abs'
            return [[h[j], h[j] * abs(self.tauJ[j])] for j in range(self.n_joints + 1)]

    def __generalized_hysteresis_model(self):
        """
        Returns the hysteresis model used in the Coulomb friction model.
        """
        assert self.hysteresis_model.lower() in ('sgn', 'gms')

        if self.hysteresis_model.lower() == 'sgn':
            return [sp.sign(self.qd[j]) for j in range(self.n_joints + 1)]
        else:
            print(f"GMS model not yet implemented - using a simple static model instead.")
            return [sp.sign(self.qd[j]) for j in range(self.n_joints + 1)]

    def __viscous_friction_basis(self):
        """
        Returns the basis (regressor) of the viscous friction model.
        """

        fv_basis = [[sp.Integer(0)] * len(self.viscous_friction_powers) for _ in range(self.n_joints + 1)]
        for j in range(self.n_joints + 1):
            for i, power in enumerate(self.viscous_friction_powers):
                if power % 2 == 0:  # even
                    fv_basis[j][i] = self.qd[j] ** (power - 1) * abs(self.qd[j])
                else:  # odd
                    fv_basis[j][i] = self.qd[j] ** power
        return fv_basis

    def basis(self):
        """
        Returns a list of 'n_joints + 1' elements with each element comprising a list of joint dynamics parameters for that
        corresponding joint.
        """

        fc_basis = self.__coulomb_friction_basis()
        fv_basis = self.__viscous_friction_basis()
        return [[*fc_basis[j], *fv_basis[j]] for j in range(self.n_joints + 1)]

    def get_dynamics(self):
        """
        Returns a list of joint dynamics with the list elements corresponding to the joint dynamics of each joint.
        """
        jd_basis = self.basis()
        jd_par = self.parameters()
        return [sum([b * p for b, p in zip(jd_basis[j], jd_par[j])]) for j in range(len(jd_basis))]
