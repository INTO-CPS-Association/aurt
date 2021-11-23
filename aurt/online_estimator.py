from abc import ABC, abstractmethod
import numpy as np
from logging import Logger

from aurt.caching import Cache
from aurt.linear_system import LinearSystem


class OnlineEstimator(ABC):
    """
    An abstract base class for an estimator for a linear system. The method requiring implementation are:\n
      - gain_matrix()
    """
    
    @abstractmethod
    def __init__(self, logger: Logger, cache: Cache, linear_system: LinearSystem, name: str=None, **kwargs):
        self.logger = logger
        self._cache = cache
        self._name = name
        self._linear_system = linear_system

        # self.initialize_system(kwargs)

        n_par = linear_system.number_of_parameters()
        self.parameter = kwargs['initial_parameter']
        self.parameters = np.zeros((sum(n_par), 1))
    
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def parameters(self) -> np.array:
        """
        An 'n_joints'-element list with each element 'j' comprising a list of base parameters related to joint 'j'.
        """
        return self._parameters
    
    @parameters.setter
    def parameters(self, parameters: np.array):
        self._parameters = parameters
    
    def update(self, measurement: np.array, states: np.array) -> np.array:
        obs_mat = self._linear_system.observation_matrix(states)
        eps = measurement - obs_mat * self.parameters  # Innovations
        self.parameters = self.parameters + self.gain_matrix(measurement, obs_mat)*eps  # Parameter estimate
        return self.parameters
    
    @abstractmethod
    def gain_matrix(self, measurement: np.array, observation_matrix: np.array) -> np.array:
        """
        The gain matrix used to update the estimate of the parameters in method 'update'.
        """
        pass
