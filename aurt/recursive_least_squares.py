import numpy as np
from logging import Logger

from aurt.caching import Cache
from aurt.linear_system import LinearSystem
from aurt.recursive_estimator import RecursiveEstimator


class RecursiveLeastSquares(RecursiveEstimator):
    """
    A Recursive Least Squares (RLS) estimator for the provided linear system.
    """
    
    def __init__(self, logger: Logger, cache: Cache, linear_system: LinearSystem, name: str=None, initial_parameter=None, **kwargs):
        super().__init__(logger, cache, linear_system, name, initial_parameter)

        n_par = linear_system.number_of_parameters()
        self.parameters = np.zeros((sum(n_par), 1))
        
        self._forgetting_factor = kwargs['forgetting']
        self._covariance_matrix = np.diag(kwargs['variance'])
    
    def gain_matrix(self, observation_matrix: np.array) -> np.array:
        """
        The kalman filter gain matrix used to update the estimate of the parameters in method 'update_parameters'.
        """
        self._kalman_gain = self._covariance_matrix @ observation_matrix.T @ np.linalg.pinv(self._forgetting_factor + observation_matrix @ self._covariance_matrix @ observation_matrix.T)
        self._covariance_matrix = (1 - self._kalman_gain @ observation_matrix) @ self._covariance_matrix / self._forgetting_factor
        return self._kalman_gain
