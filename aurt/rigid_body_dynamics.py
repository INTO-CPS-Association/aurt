import sympy as sp

from aurt.file_system import cache_object


class RigidBodyDynamics:
    def __init__(self, n_joints, modified_dh, gravity=None):
        self.mdh = modified_dh
        self.n_joints = n_joints

        if gravity is None:
            print(f"No gravity direction was specified. Assuming the first joint axis of rotation to be parallel to gravity...")
            gravity = [0, 0, -9.81]
        self.gravity = gravity

    def parameters(self):
        """
        Returns a list of 'n_joints + 1' elements with each element comprising a list of all rigid body parameters related to
        that corresponding link.
        """
        m = sp.symbols([f"m{j}" for j in range(self.n_joints + 1)])
        mX = sp.symbols([f"mX{j}" for j in range(self.n_joints + 1)])
        mY = sp.symbols([f"mY{j}" for j in range(self.n_joints + 1)])
        mZ = sp.symbols([f"mZ{j}" for j in range(self.n_joints + 1)])
        XX = sp.symbols([f"XX{j}" for j in range(self.n_joints + 1)])
        XY = sp.symbols([f"XY{j}" for j in range(self.n_joints + 1)])
        XZ = sp.symbols([f"XZ{j}" for j in range(self.n_joints + 1)])
        YY = sp.symbols([f"YY{j}" for j in range(self.n_joints + 1)])
        YZ = sp.symbols([f"YZ{j}" for j in range(self.n_joints + 1)])
        ZZ = sp.symbols([f"ZZ{j}" for j in range(self.n_joints + 1)])

        return [[XX[j], XY[j], XZ[j], YY[j], YZ[j], ZZ[j], mX[j], mY[j], mZ[j], m[j]] for j in range(self.n_joints + 1)]

    def number_of_parameters_each_rigid_body(self):
        return len(self.parameters()[0])

    def basis(self):
        self.f=1

    def __replace_first_moments(self, args):

        tau_sym_j = args[0][0]  # tau_sym[j]
        j = args[0][1]
        m = args[1]  # [m1, ..., mN]
        PC = args[2]  # PC = [PC[1], ..., PC[N]], PC[j] = [PCxj, PCyj, PCzj]
        m_pc = args[3]  # mPC = [mPC[1], ..., mPC[N]], mPC[j] = [mXj, mYj, mZj]

        # TODO: REMOVE THE 'IGNORE J=0' STUFF
        if j == 0:
            print(f"Ignore joint {j}...")
            return tau_sym_j

        n_cartesian = len(m_pc[0])
        for jj in range(j, self.n_joints+1):  # The joint j torque equation is affected by dynamic coefficients only for links jj >= j
            for i in range(n_cartesian):  # Cartesian coordinates loop
                # Print progression counter
                total = (self.n_joints+1-j)*n_cartesian
                progress = (jj - j)*n_cartesian + i + 1
                print(f"Task {j}: {progress}/{total} (tau[{j}]: {m[jj]}*{PC[jj][i]} -> {m_pc[jj][i]})")
                tau_sym_j = tau_sym_j.expand().subs(m[jj] * PC[jj][i], m_pc[jj][i])  # expand() is - for unknown reasons - needed for subs() to consistently detect the "m*PC" products

        return tau_sym_j

    def get_dynamics(self):
        rbd = cache_object('./rigid_body_dynamics',
                                   lambda: compute_torques_symbolic_ur(q, qd, qdd, f_tcp, n_tcp, i_cor, g,
                                                                       inertia_is_wrt_CoM=False))

        js = list(range(self.n_joints + 1))
        dynamics_per_task = [rbd[j] for j in js]  # Allows one to control how many tasks by controlling how many js.
        data_per_task = list(product(zip(dynamics_per_task, js), [m], [PC], [mPC]))

        with Pool() as p:
            rbd_linearizable = p.map(self.__replace_first_moments, data_per_task)

        return rbd_linearizable
