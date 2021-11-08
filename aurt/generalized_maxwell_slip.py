import numpy as np
import sympy as sp

class GeneralizedMaxwellSlip():
    r_Delta = 4
    r_k = 2

    def __init__(self, backlash, n_elements=4):
        self._backlash = backlash
        self._n_elements = n_elements

        self._z = np.zeros((n_elements, 1))
        self._Delta = np.array((n_elements, 1), None)
        self._k = np.array((n_elements, 1), None)

        self._init_sys()
    
    def _init_sys(self):
        sum_i_rk = 0
        for i in range(self._n_elements):
            sum_i_rk += i**GeneralizedMaxwellSlip.r_k
        
        for i in range(self._n_elements):
            self._Delta[i] = (i/self._n_elements)**GeneralizedMaxwellSlip.r_Delta
            self._k[i] = (1/self._backlash)*((self._n_elements + 1 - i)**GeneralizedMaxwellSlip.r_k)/sum_i_rk
    
    def _state_update(self, dx):
        self._z = np.sign(dx + self._z) * min(abs(dx + self._z), self._Delta)
        return self._z

    def state_sym(self):
        dx = sp.symbols("dx")
        z = sp.symbols(f"z:{self._n_elements}")
        Delta = sp.symbols(f"Delta:{self._n_elements}")
        z = sp.zeros(self._n_elements, 1)
        
        for i in range(self._n_elements):
            z[i] = sp.sign(dx + z[i]) * sp.Min(abs(dx + z[i]), Delta[i])
    
    def update(self, dx):
        self._state_update(dx)
        f = self._k.T * self._z
        return f