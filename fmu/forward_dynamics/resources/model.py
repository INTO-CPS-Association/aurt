import pickle
import numpy as np
from scipy.integrate import solve_ivp

class Simulator:
    step_size = 0.5e-3

    def __init__(self, calibrated_robot, tau, q, qd, gravity) -> None:
        
        self.calibrated_robot = calibrated_robot
        self.tolerance = None
        self.tau = tau
        self.q = q
        self.qd = qd
        self.gravity = gravity

    def simulate(self, simulation_time):
        
        n_joints = self.calibrated_robot.robot_dynamics.n_joints
        def f(t, y):
            q = y[:n_joints]
            qd = y[n_joints:]
            qdd = self.calibrated_robot.angular_acceleration(self.tau, q, qd, self.gravity)
            return np.concatenate((qd, qdd))
        
        y0 = np.concatenate((self.q, self.qd))
        self.tolerance = 1e90  # manually overwrite tolerance to fix the step size
        
        sol = solve_ivp(f, (0, simulation_time), y0, first_step=Simulator.step_size, max_step=Simulator.step_size, atol = self.tolerance, rtol=self.tolerance)
        self.q[:] = sol.y[:n_joints, -1]
        self.qd[:] = sol.y[n_joints:, -1]

class Model:
    max_joints = 10
    
    def __init__(self) -> None:
        with open("./rd_calibrated.pickle",'rb') as f:
            self.calibrated_robot = pickle.load(f)
        self.n_joints = self.calibrated_robot.robot_dynamics.n_joints

        self.all_variables = np.zeros((5*Model.max_joints + 3,))
        self.tau = self.all_variables[:self.n_joints]
        self.q = self.all_variables[Model.max_joints:Model.max_joints + self.n_joints]
        self.qd = self.all_variables[2*Model.max_joints:2*Model.max_joints + self.n_joints]
        self.q_init = self.all_variables[3*Model.max_joints:3*Model.max_joints + self.n_joints]
        self.qd_init = self.all_variables[4*Model.max_joints:4*Model.max_joints + self.n_joints]
        self.gravity = self.all_variables[5*Model.max_joints:5*Model.max_joints + 3]

        self.simulator = Simulator(self.calibrated_robot, self.tau, self.q, self.qd, self.gravity)

    def fmi2DoStep(self, current_time, step_size, no_step_prior):
        self.simulator.simulate(step_size)
        return Fmi2Status.ok

    def fmi2EnterInitializationMode(self):
        return Fmi2Status.ok

    def fmi2ExitInitializationMode(self):
        self.q[:] = self.q_init
        self.qd[:] = self.qd_init

        return Fmi2Status.ok

    def fmi2SetupExperiment(self, start_time, stop_time, tolerance):
        self.simulator.tolerance = tolerance
        return Fmi2Status.ok

    def fmi2SetReal(self, references, values):
        return self._set_value(references, values)

    def fmi2SetInteger(self, references, values):
        return self._set_value(references, values)

    def fmi2SetBoolean(self, references, values):
        return self._set_value(references, values)

    def fmi2SetString(self, references, values):
        return self._set_value(references, values)

    def fmi2GetReal(self, references):
        return self._get_value(references)

    def fmi2GetInteger(self, references):
        return self._get_value(references)

    def fmi2GetBoolean(self, references):
        return self._get_value(references)

    def fmi2GetString(self, references):
        return self._get_value(references)

    def fmi2Reset(self):
        return Fmi2Status.ok

    def fmi2Terminate(self):
        return Fmi2Status.ok

    def fmi2ExtSerialize(self):

        bytes = pickle.dumps(self.all_variables)
        return Fmi2Status.ok, bytes

    def fmi2ExtDeserialize(self, bytes) -> int:

        self.all_variables = pickle.loads(bytes)

        return Fmi2Status.ok

    def _set_value(self, references, values):
        
        for r, v in zip(references, values):
            self.all_variables[r] = v

        return Fmi2Status.ok

    def _get_value(self, references):

        values = []

        for r in references:
            values.append(self.all_variables[r])

        return Fmi2Status.ok, values


class Fmi2Status:
    """Represents the status of the FMU or the results of function calls.

    Values:
        * ok: all well
        * warning: an issue has arisen, but the computation can continue.
        * discard: an operation has resulted in invalid output, which must be discarded
        * error: an error has ocurred for this specific FMU instance.
        * fatal: an fatal error has ocurred which has corrupted ALL FMU instances.
        * pending: indicates that the FMu is doing work asynchronously, which can be retrived later.

    Notes:
        FMI section 2.1.3

    """

    ok = 0
    warning = 1
    discard = 2
    error = 3
    fatal = 4
    pending = 5
    
    
if __name__ == "__main__":
    # Initialize model
    model = Model()
    model.tau[:] = np.zeros((model.n_joints,))
    model.q[:] = np.zeros((model.n_joints,))
    model.qd[:] = np.zeros((model.n_joints,))
    model.q_init[:] = np.zeros((model.n_joints,))
    model.qd_init[:] = np.zeros((model.n_joints,))
    model.gravity[:] = np.array([0, 0, -9.81])

    # Simulate
    import time
    t0 = time.time()
    model.fmi2SetupExperiment(0.0, 1.0, 1e2)
    model.fmi2EnterInitializationMode()
    model.fmi2ExitInitializationMode()
    t1 = time.time()
    model.fmi2DoStep(0.0, 0.05, False)
    t2 = time.time()
    print(f"Initialization time: {t1 - t0} s.")
    print(f"doStep time: {t2 - t1} s.")
