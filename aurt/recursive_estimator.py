from abc import ABC, abstractmethod
import numpy as np
from logging import Logger

from aurt.caching import Cache
from aurt.linear_system import LinearSystem


class RecursiveEstimator(ABC):
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

        self.parameters = kwargs['initial_parameter']
    
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
    
    def update_parameters(self, measurement: np.array, states: np.array) -> np.array:
        observation_matrix = self._linear_system.observation_matrix(states)
        eps = measurement - observation_matrix * self.parameters  # Innovations
        self.parameters = self.parameters + self.gain_matrix(measurement, observation_matrix) * eps  # Parameter estimate
        return self.parameters
    
    @abstractmethod
    def gain_matrix(self, observation_matrix: np.array) -> np.array:
        """
        The gain matrix used to update the estimate of the parameters in method 'update'.
        """
        pass
