import sys
import sympy as sp
import numpy as np


class JointDynamics:
    def __init__(self, n_joints, load_model=None, hysteresis_model=None, viscous_powers=None):
        self.n_joints = n_joints
        self.__qd = [sp.Integer(0)] + [sp.symbols(f"qd{j}") for j in range(1, self.n_joints + 1)]
        self.__tauJ = sp.symbols([f"tauJ{j}" for j in range(self.n_joints + 1)])

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
        return [[*fcs[j], *fvs[j]] for j in range(1, self.n_joints + 1)]

    def number_of_parameters(self):
        par = self.parameters()
        return [len(par[j]) for j in range(self.n_joints)]

    def __coulomb_friction_basis(self):
        """
        Returns the basis (regressor) of the Coulomb friction model.
        """

        assert self.load_model.lower() in ('none', 'square', 'abs')

        h = self.__generalized_hysteresis_model()
        if self.load_model.lower() == 'none':
            return [[h[j]] for j in range(self.n_joints + 1)]
        elif self.load_model.lower() == 'square':
            return [[h[j], h[j] * self.__tauJ[j] ** 2] for j in range(self.n_joints + 1)]
        else:  # load_model == 'abs'
            return [[h[j], h[j] * abs(self.__tauJ[j])] for j in range(self.n_joints + 1)]

    def __generalized_hysteresis_model(self):
        """
        Returns the hysteresis model used in the Coulomb friction model.
        """
        assert self.hysteresis_model.lower() in ('sign', 'gms')

        if self.hysteresis_model.lower() == 'sign':
            return [sp.sign(self.__qd[j]) for j in range(self.n_joints + 1)]
        else:
            print(f"GMS model not yet implemented - using a simple static model instead.")
            return [sp.sign(self.__qd[j]) for j in range(self.n_joints + 1)]

    def __viscous_friction_basis(self):
        """
        Returns the basis (regressor) of the viscous friction model.
        """

        fv_basis = [[sp.Integer(0)] * len(self.viscous_friction_powers) for _ in range(self.n_joints + 1)]
        for j in range(self.n_joints + 1):
            for i, power in enumerate(self.viscous_friction_powers):
                if power % 2 == 0:  # even
                    fv_basis[j][i] = self.__qd[j] ** (power - 1) * abs(self.__qd[j])
                else:  # odd
                    fv_basis[j][i] = self.__qd[j] ** power
        return fv_basis

    def observation_matrix_joint(self, j, qd_j_num, tauJ_j_num):
        """Returns a (n_samples x self.number_of_parameters[j]) observation matrix"""
        assert qd_j_num.size == tauJ_j_num.size
        assert 0 <= j < self.n_joints

        n_samples = qd_j_num.size
        args_sym = [self.__qd[j+1], self.__tauJ[j+1]]
        args_num = np.concatenate((qd_j_num[:, np.newaxis], tauJ_j_num[:, np.newaxis]), axis=1).transpose()
        regressor_j = self.regressor()[j]
        observation_matrix_j = np.zeros((n_samples, regressor_j.shape[1]))
        sys.setrecursionlimit(int(1e6))
        reg_j_fcn = sp.lambdify(args_sym, regressor_j, 'numpy')
        observation_matrix_j[:n_samples, :] = reg_j_fcn(*args_num).squeeze().transpose()

        return observation_matrix_j

    def observation_matrix(self, qd_num, tauJ_num):
        """Returns a list of observation matrices for each joint."""
        assert qd_num.shape == tauJ_num.shape

        return [self.observation_matrix_joint(j, qd_num[j, :], tauJ_num[j, :]) for j in range(self.n_joints)]

    def regressor(self):
        """
        Returns an 'n_joints' list of regressors.
        """

        fc_basis = self.__coulomb_friction_basis()
        fv_basis = self.__viscous_friction_basis()
        return [sp.Matrix([*fc_basis[j], *fv_basis[j]]).T for j in range(1, self.n_joints + 1)]

    def dynamics(self):
        """
        Returns a list of joint dynamics with the list elements corresponding to the joint dynamics of each joint.
        """
        jd_basis = self.regressor()
        jd_par = self.parameters()
        return [sum([b * p for b, p in zip(jd_basis[i], jd_par[i])]) for i in range(len(jd_basis))]
