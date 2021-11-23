import numpy as np
from logging import Logger

from aurt.caching import Cache
from aurt.linear_system import LinearSystem
from aurt.recursive_estimator import RecursiveEstimator


class RecursiveLeastSquares(RecursiveEstimator):
    """
    A Recursive Least Squares (RLS) online estimator.
    """
    
    def __init__(self, logger: Logger, cache: Cache, linear_system: LinearSystem, name: str=None, **kwargs):
        super().__init__(logger, cache, linear_system, name, kwargs)

        n_par = linear_system.number_of_parameters()
        self.parameter = kwargs['initial_parameter']
        self.parameters = np.zeros((sum(n_par), 1))
        
        self._forgetting_factor = kwargs['forgetting']
        self._covariance_matrix = np.diag(kwargs['variance'])
    
    def gain_matrix(self, observation_matrix: np.array) -> np.array:
        """
        The kalman filter gain matrix used to update the estimate of the parameters in method 'update'.
        """
        self._kalman_gain = self._covariance * observation_matrix * np.linalg.inv(self._forgetting_factor + observation_matrix.T * self._covariance * observation_matrix)
        self._covariance = (1 - self._kalman_gain * observation_matrix.T) * self._covariance_matrix / self._forgetting_factor
        return self._kalman_gain
