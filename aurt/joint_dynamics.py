import sys
import sympy as sp
import numpy as np
from logging import Logger

from aurt.caching import Cache
from aurt.linear_system import LinearSystem
from aurt.generalized_maxwell_slip import GeneralizedMaxwellSlip


class JointDynamics(LinearSystem):
    def __init__(self, logger: Logger, cache: Cache, n_joints: int, load_model=None, hysteresis_model=None, viscous_powers=None):
        super().__init__(logger, cache, name="joint dynamics")

        self.n_joints = n_joints
        self._qd = [sp.Integer(0)] + [sp.symbols(f"qd{j}") for j in range(1, self.n_joints + 1)]
        self._tauJ = sp.symbols([f"tauJ{j}" for j in range(self.n_joints + 1)])

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

        super().compute_linearly_independent_system()

    def states(self):
        return [list(a) for a in zip(self._qd[1:], self._tauJ[1:])]

    def _parameters_full(self):
        """
        Returns a list of 'n_joints + 1' elements with each element comprising a list of joint dynamics parameters for that
        corresponding joint.
        """

        fcs = self._coulomb_friction_parameters()
        fvs = self._viscous_friction_parameters()
        return [[*fcs[j], *fvs[j]] for j in range(1, self.n_joints + 1)]

    def _coulomb_friction_parameters(self):
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

    def _viscous_friction_parameters(self):
        """
        Returns a list of 'Njoints + 1' elements, each element comprising a list of viscous friction parameters for each
        joint.
        """

        return [[sp.symbols(f"Fv{j}_{i}") for i in self.viscous_friction_powers] for j in range(self.n_joints + 1)]

    def _coulomb_friction_basis(self):
        """
        Returns the basis (regressor) of the Coulomb friction model.
        """

        assert self.load_model.lower() in ('none', 'square', 'abs')

        h = self._generalized_hysteresis_model()
        if self.load_model.lower() == 'none':
            return [[h[j]] for j in range(self.n_joints + 1)]
        elif self.load_model.lower() == 'square':
            return [[h[j], h[j]*self._tauJ[j]**2] for j in range(self.n_joints + 1)]
        else:  # load_model == 'abs'
            return [[*h[j], *(sp.Matrix(h[j])*abs(self._tauJ[j])).tolist()] for j in range(self.n_joints + 1)]

    def _generalized_hysteresis_model(self):
        """
        Returns the hysteresis model used in the Coulomb friction model.
        """
        assert self.hysteresis_model.lower() in ('sign', 'gms')

        if self.hysteresis_model.lower() == 'sign':
            return [sp.sign(self._qd[j]) for j in range(self.n_joints + 1)]
        else:
            print(f"GMS model not yet implemented - using a simple static model instead.")
            n_gms_elements = 4

            dq = sp.symbols(f"dq:{self.n_joints + 1}")
            z0 = [sp.Matrix([sp.Symbol('z%d%d'%(j, i)) for i in range(n_gms_elements)]) for j in range(self.n_joints + 1)]  # Initialization of an 'n_joints'-elements list, each element equal to [z{0}, ..., z{n_gms_elements}] (type sp.Matrix)
            Delta = [sp.Matrix([sp.Symbol('Delta%d%d'%(j, i)) for i in range(n_gms_elements)]) for j in range(self.n_joints + 1)]
            
            z1 = z0.copy()
            for j in range(self.n_joints + 1):
                for i in range(n_gms_elements):
                    z1[j][i] = sp.sign(dq[j] + z0[j][i]) * sp.Min(abs(dq[j] + z0[j][i]), Delta[j][i])
                z1[j] = z1[j].tolist()
            
            return z1

    def _viscous_friction_basis(self):
        """
        Returns the basis (regressor) of the viscous friction model.
        """

        fv_basis = [[sp.Integer(0)] * len(self.viscous_friction_powers) for _ in range(self.n_joints + 1)]
        for j in range(self.n_joints + 1):
            for i, power in enumerate(self.viscous_friction_powers):
                if power % 2 == 0:  # even
                    fv_basis[j][i] = self._qd[j] ** (power - 1) * abs(self._qd[j])
                else:  # odd
                    fv_basis[j][i] = self._qd[j] ** power
        return fv_basis

    def _regressor_joint_parameters_for_joint(self, j, par_j):
        """
        Those columns of row 'j' of the regressor related to the parameters 'par_j' for joint j.
        We make use of the fact that the parameters for any joint j affects the torque of joint j only.
        This also means that the joints dynamics regressor is block diagonal.
        """
        regressor_j_par_j = sp.zeros(1, self._number_of_parameters_full()[par_j])

        if j == par_j:
            fc_basis_j = self._coulomb_friction_basis()[j + 1]
            fv_basis_j = self._viscous_friction_basis()[j + 1]
            regressor_j_par_j = sp.Matrix([*fc_basis_j, *fv_basis_j]).T
        return regressor_j_par_j
