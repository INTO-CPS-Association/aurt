import pathlib

import sympy as sp
from multiprocessing import Pool
from itertools import product

from aurt.file_system import cache_object, project_root
from aurt.num_sym_layers import spvector
from aurt.globals import get_ur_frames
from aurt.torques import compute_torques_symbolic_ur


class RigidBodyDynamics:
    def __init__(self, modified_dh, n_joints, gravity=None):
        self.mdh = modified_dh
        self.n_joints = n_joints

        # TODO: Change such that q, qd, and qdd are generated based on 'mdh'
        self.q = [sp.Integer(0)] + [sp.symbols(f"q{j}") for j in range(1, self.n_joints + 1)]
        self.qd = [sp.Integer(0)] + [sp.symbols(f"qd{j}") for j in range(1, self.n_joints + 1)]
        self.qdd = [sp.Integer(0)] + [sp.symbols(f"qdd{j}") for j in range(1, self.n_joints + 1)]

        self.m = sp.symbols([f"m{j}" for j in range(self.n_joints + 1)])
        self.mX = sp.symbols([f"mX{j}" for j in range(self.n_joints + 1)])
        self.mY = sp.symbols([f"mY{j}" for j in range(self.n_joints + 1)])
        self.mZ = sp.symbols([f"mZ{j}" for j in range(self.n_joints + 1)])
        self.pc = get_ur_frames(None, spvector)
        self.m_pc = [[self.mX[j], self.mY[j], self.mZ[j]] for j in range(self.n_joints + 1)]

        self.XX = sp.symbols([f"XX{j}" for j in range(self.n_joints + 1)])
        self.XY = sp.symbols([f"XY{j}" for j in range(self.n_joints + 1)])
        self.XZ = sp.symbols([f"XZ{j}" for j in range(self.n_joints + 1)])
        self.YY = sp.symbols([f"YY{j}" for j in range(self.n_joints + 1)])
        self.YZ = sp.symbols([f"YZ{j}" for j in range(self.n_joints + 1)])
        self.ZZ = sp.symbols([f"ZZ{j}" for j in range(self.n_joints + 1)])
        self.i_cor = [sp.zeros(3, 3) for i in range(self.n_joints + 1)]
        for j in range(self.n_joints + 1):
            self.i_cor[j] = sp.Matrix([
                [self.XX[j], self.XY[j], self.XZ[j]],
                [self.XY[j], self.YY[j], self.YZ[j]],
                [self.XZ[j], self.YZ[j], self.ZZ[j]]
            ])

        fx, fy, fz, nx, ny, nz = sp.symbols(f"fx fy fz nx ny nz")
        self.f_tcp = sp.Matrix([fx, fy, fz])  # Force at the TCP
        self.n_tcp = sp.Matrix([nx, ny, nz])  # Moment at the TCP

        if gravity is None:
            print(f"No gravity direction was specified. Assuming the first joint axis of rotation to be parallel to gravity...")
            gravity = [0, 0, -9.81]
        self.gravity = gravity

    def parameters(self):
        """
        Returns a list of 'n_joints + 1' elements with each element comprising a list of all rigid body parameters related to
        that corresponding link.
        """

        return [[self.XX[j], self.XY[j], self.XZ[j], self.YY[j], self.YZ[j], self.ZZ[j],
                 self.mX[j], self.mY[j], self.mZ[j], self.m[j]] for j in range(self.n_joints + 1)]

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
        def compute_dynamics_and_replace_first_moments():
            rbd = cache_object('./rigid_body_dynamics',
                               lambda: compute_torques_symbolic_ur(self.q, self.qd, self.qdd, self.f_tcp, self.n_tcp,
                                                                   self.i_cor, self.gravity,
                                                                   inertia_is_wrt_CoM=False))
            js = list(range(self.n_joints + 1))
            dynamics_per_task = [rbd[j] for j in js]  # Allows one to control how many tasks by controlling how many js.
            data_per_task = list(product(zip(dynamics_per_task, js), [self.m], [self.pc], [self.m_pc]))

            with Pool() as p:
                rbd_linearizable = p.map(self.__replace_first_moments, data_per_task)
            return rbd_linearizable

        rbd_linearizable = cache_object(pathlib.Path.joinpath(project_root(), 'cache', 'rigid_body_dynamics'),
                                        compute_dynamics_and_replace_first_moments)

        return rbd_linearizable
