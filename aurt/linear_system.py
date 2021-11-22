from abc import ABC, abstractmethod
from math import ceil
import sympy as sp
import numpy as np
import sys
from logging import Logger
from itertools import chain
from inspect import signature

from aurt.caching import Cache
from aurt.dynamics_aux import list_2D_to_sympy_vector


class LinearSystem(ABC):
    """
    An abstract base class for a linear system. The methods requiring implementation are:\n
      - states()
      - _parameters_full()
      - _regressor_joint_parameters_for_joint()
    """
    
    @abstractmethod
    def __init__(self, logger: Logger, cache: Cache, name: str=None):
        self.logger = logger
        self._cache = cache
        self._name = name

        self._parameters = None
        self.is_base_parameter = None
    
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, name: str):
        self._name = name
    
    @abstractmethod
    def states(self) -> list:
        """
        A list of states for the system.
        """
        pass

    @abstractmethod
    def _parameters_full(self) -> list:
        """
        An 'n_joints'-element list with each element 'j' comprising a list of parameters related to joint 'j'.
        """
        pass

    def _number_of_parameters_full(self) -> list:
        """Returns a list of elements with each element 'j' indicating the number of parameters related to joint 'j'."""
        par = self._parameters_full()
        return [len(par[j]) for j in range(len(par))]

    @property
    def parameters(self) -> list:
        """
        An 'n_joints'-element list with each element 'j' comprising a list of base parameters related to joint 'j'.
        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters

    def number_of_parameters(self) -> list:
        """Returns a list of elements with each element indicating the number of Base Parameters (BP) for that joint."""
        par = self.parameters
        return [len(par[j]) for j in range(len(par))]

    def _regressor(self) -> sp.Matrix:
        # Construct the regressor matrix with possibly linearly dependent columns
        n_par = self._number_of_parameters_full()
        n_joints = len(n_par)
        regressor = sp.zeros(n_joints, sum(n_par))
        for j in range(n_joints):
            regressor[j, :] = self._regressor_joint(j)
        
        # Return regressor matrix having possibly linearly dependent columns
        return regressor
    
    def regressor(self) -> sp.Matrix:
        # Construct the regressor matrix with linearly independent columns
        n_par = self.number_of_parameters()
        n_joints = len(n_par)
        regressor = sp.zeros(n_joints, sum(n_par))
        for j in range(n_joints):
            regressor[j, :] = self.regressor_joint(j)
        
        return regressor

    def _regressor_joint(self, j: int) -> sp.Matrix:
        """Constructs row 'j' of the regressor matrix."""
        n_par = self._number_of_parameters_full()

        reg_j = sp.zeros(1, sum(n_par))
        for par_j in range(self.n_joints):
            col_start = sum(n_par[:par_j])
            col_end = col_start + n_par[par_j]
            reg_j[:, col_start:col_end] = self._regressor_joint_parameters_for_joint(j, par_j)
        
        return reg_j

    def regressor_joint(self, j: int) -> sp.Matrix:
        """Constructs row 'j' of the regressor matrix."""
        def compute_regressor_joint():
            n_par = self.number_of_parameters()

            reg_j = sp.zeros(1, sum(n_par))
            for par_j in range(self.n_joints):
                col_start = sum(n_par[:par_j])
                col_end = col_start + n_par[par_j]
                reg_j[:, col_start:col_end] = self.regressor_joint_parameters_for_joint(j, par_j)
            return reg_j
        
        # filename = f"{self.name}_regressor_j{j}"
        reg_j =  compute_regressor_joint()  # self._cache.get_or_cache(filename, compute_regressor_joint)

        return reg_j
    
    @abstractmethod
    def _regressor_joint_parameters_for_joint(self, j: int, par_j: int) -> sp.Matrix:
        """
        Those regressors for joint 'j' corresponding to the parameters related to joint 'par_j'.
        The output of this method must be an m-element row vector with 'm' equal to the 'par_j'th element
        of the output from method 'number_of_parameters()'.
        """
        pass

    def regressor_joint_parameters_for_joint(self, j: int, par_j: int) -> sp.Matrix:
        filename = f"{self.name}_regressor_full_j={j}_par_j={par_j}"
        reg_full_j_parj = self._cache.get_or_cache(filename, lambda: self._regressor_joint_parameters_for_joint(j, par_j))
        return reg_full_j_parj[:, self.is_base_parameter[par_j]]

    def compute_linearly_independent_system(self):
        """
        Computes the indices for the set of linearly independent columns of the regressor matrix corresponding to 
        the set of Base Parameters (BP) for the linear system. The number of BP is obtained as the maximum 
        obtainable rank of the observation matrix using a set of randomly generated dummy observations for the 
        states of the system. The specific indices for the BP are obtained as the indices of the set of linearly 
        independent columns of the observation matrix. This is done by conducting a QR decomposition of the
        observation matrix and analyzing the diagonal elements of the upper triangular (R) matrix. Those diagonal
        elements of R with an absolute value larger than some numerical threshold are considered linearly
        independent.
        """
        # 1. Compute the number of base parameters
        n_rank_convergence = 5  # No. of iterations in which the rank should be non-increasing for the procedure to be considered convergent
        max_number_of_rank_evaluations = 200  # Maximum no. of iterations in while-loop
        qr_numerical_threshold = 1e-12  # Threshold value for the diagonal elements of the R matrix to be considered linearly independent

        states1D = list(chain.from_iterable(self.states()))
        sys.setrecursionlimit(int(1e6))  # Prevents errors in sympy lambdify
        regressor_full = self._regressor()
        regressor_lambdified = sp.lambdify(states1D, regressor_full, 'numpy')
        min_rank_evals = ceil(max(n_rank_convergence, regressor_full.shape[1]/regressor_full.shape[0]*1.5))  # Minimum no. of rank evaluations

        n_args = len(signature(regressor_lambdified).parameters)  # No. of arguments to the lambdified regressor

        # 1.1 Initialization
        dummy_args_init = np.random.uniform(low=-np.pi, high=np.pi, size=n_args)
        obs_mat = regressor_lambdified(*dummy_args_init)
        n_rank_evals = 0  # Loop counter for observation matrix concatenations
        rank_obs_mat = []

        # 1.2 Compute the rank of the system iteratively, exit upon convergence or when the maximum number of iterations has been reached
        self.logger.debug(f"Determining numerically the number of base parameters of '{self.name}'...")
        while n_rank_evals < min_rank_evals or rank_obs_mat[-1] > rank_obs_mat[-n_rank_convergence]:  # While the rank of the observation matrix keeps increasing (converging)
            n_rank_evals += 1

            dummy_args = np.random.uniform(low=-np.pi, high=np.pi, size=n_args)
            obs_mat_i = regressor_lambdified(*dummy_args)
            obs_mat = np.append(obs_mat, obs_mat_i, axis=0)
            rank_obs_mat.append(np.linalg.matrix_rank(obs_mat))

            self.logger.debug(f"No. of base parameters for '{self.name}' at iteration {n_rank_evals}: {rank_obs_mat[-1]}")

            if n_rank_evals > max_number_of_rank_evaluations:
                self.logger.error(f"The iterative computation of the no. of base parameters did not converge within the maximum ({n_rank_evals}) allowed number of iterations.")
                break
        
        # 2. Compute the actual base parameters
        r = np.linalg.qr(obs_mat, mode='r')
        idx_is_base = abs(np.diag(r)) > qr_numerical_threshold*1e6  # List of bools, e.g. [True, False, True, ...], indicating if some term is part of the minimal basis
        idx_base_parameters = np.where(idx_is_base)[0].tolist()  # List of (integer) indices
        assert len(idx_base_parameters) == rank_obs_mat[-1], f"The rank was estimated to be {rank_obs_mat[-1]}, however the no. of base parameters was {len(idx_base_parameters)}. These should be equal."

        # 3. Compute at the joint level; 1) the number of base parameters and 2) the base parameters  
        parameters_full_1D = list(chain.from_iterable(self._parameters_full()))
        parameters_base_1D = [parameters_full_1D[i] for i in idx_base_parameters]

        # 3.1 Initialization
        number_of_parameters_full = self._number_of_parameters_full()  # No. of parameters in user-defined (possibly linearly dependent) system
        number_of_parameters = number_of_parameters_full.copy()  # No. of (base) parameters in minimal (linearly independent) system
        parameters = self._parameters_full().copy()  # Parameters of user-defined (possibly linearly dependent) system
        n_joints = len(number_of_parameters_full)
        is_base_parameter = [[True for _ in range(number_of_parameters_full[j])] for j in range(n_joints)]

        # 3.2 Eliminate linearly dependent parameters
        for j in range(n_joints):
            for i in reversed(range(number_of_parameters_full[j])):
                if parameters[j][i].is_zero or not parameters[j][i] in parameters_base_1D:
                    number_of_parameters[j] -= 1
                    is_base_parameter[j][i] = False
                    del parameters[j][i]

        assert sum(number_of_parameters) == len(list(chain.from_iterable(parameters)))

        self._parameters = parameters
        self.is_base_parameter = is_base_parameter

        # Output information to logger
        if not all(list(chain.from_iterable(is_base_parameter))):
            self.logger.info(f"For '{self.name}' some terms were linearly dependent, thus the following parameter(s) were eliminated: {not parameters_full_1D in parameters_base_1D}")
        else:
            self.logger.info(f"'{self.name}' consist entirely of linearly independent terms, thus no parameters were eliminated.")

    def observation_matrix(self, states_num: np.array) -> np.array:
        """Constructs the observation matrix by evaluating the regressor in the data provided."""

        n_par = self.number_of_parameters()
        n_joints = len(n_par)
        n_samples = states_num.shape[1]

        observation_matrix = np.empty(n_samples*n_joints, sum(n_par))
        for j in range(n_joints):
            observation_matrix[j*n_samples:(j+1)*n_samples, :] = self.observation_matrix_joint(j, states_num)

        return observation_matrix
    
    def observation_matrix_joint(self, j: int, states_num: np.array) -> np.array:
        """
        Constructs the observation matrix for joint 'j' by evaluating the regressor for joint 'j'
        (the j'th row of the regressor matrix) in the provided data. The data should consist of a
        numpy.array with states (see the method states() for a description hereof) along axis 0 
        and time along axis 1.
        """

        n_samples = states_num.shape[1]
        n_par = self.number_of_parameters()

        observation_matrix_j = np.empty((n_samples, sum(n_par)))
        for par_j in range(self.n_joints):
            col_start = sum(n_par[:par_j])
            col_end = col_start + n_par[par_j]
            observation_matrix_j[:, col_start:col_end] = self.observation_matrix_joint_parameters_for_joint(j, par_j, states_num)
        return observation_matrix_j

    def observation_matrix_joint_parameters_for_joint(self, j: int, par_j: int, states_num: np.array) -> np.array:
        states_1D = list(chain.from_iterable(self.states()))

        assert len(states_1D) == states_num.shape[0], f"The provided argument 'states_num' has a dimension of {states_num.shape[0]} along axis 0 should have that dimension equal to the number of states ({len(states_1D)})."

        n_samples = states_num.shape[1]
        n_par_j = self.number_of_parameters()[par_j]

        regressor_j_parj = self.regressor_joint_parameters_for_joint(j, par_j)
        observation_matrix_j_parj = np.zeros((n_samples, n_par_j))
        nonzeros = [not elem.is_zero for elem in regressor_j_parj]
        if any(nonzeros):
            sys.setrecursionlimit(int(1e6))  # Prevents errors in sympy lambdify
            regressor_j_parj_nonzeros_fcn = sp.lambdify(states_1D, regressor_j_parj[:, nonzeros], 'numpy')
            observation_matrix_j_parj[:, nonzeros] = regressor_j_parj_nonzeros_fcn(*states_num).T[:, :, 0]  # "[:, :, 0]" eliminates the last (excess) dimension
        return observation_matrix_j_parj

    def dynamics(self) -> sp.Matrix:
        """The dynamics of the linear system."""
        return self.regressor() @ list_2D_to_sympy_vector(self.parameters)
