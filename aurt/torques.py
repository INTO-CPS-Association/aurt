from aurt.globals import Njoints, get_ur3e_parameters, get_ur3e_PC, get_ur5e_parameters, get_ur5e_PC, \
    get_ur_parameters_symbolic, get_ur_frames, get_P
from aurt.kinematics import get_forward_kinematics

from aurt.num_sym_layers import *


def torque_algorithm_factory_from_parameters(ur_parameter_function, ur_frames_fun):
    def create_torque_algorithm(vector, matrix, zeros_array, zeros_matrix, cos, sin, cross, dot, identity):

        identity_3 = identity(3)

        def compute_torques(q, qd, qdd, f_tcp, n_tcp, inertia, g, inertia_is_wrt_CoM=True):
            """
            Inputs: q, qd, qdd:             Joint angular positions and their first and second-order time derivatives.
                    f_tcp, n_tcp:           Force/torque at the tool center point (tcp)
                    inertia:                inertia tensor (3 x 3 matrix for each joint)
                    g:                      gravity vector (3 x 1 vector, e.g [0, 0, 9.81])
                    inertia_is_wrt_CoM:     flag to indicate whether or not the reference frame for the provided inertia
                                            is located at the center of mass (CoM) or if not, it is assumed located at
                                            the joint axes of rotation.

            Follows algorithm described in
            Craig, John J. 2009. Introduction to Robotics: Mechanics and Control, 3/E. Pearson Education India.
            """

            # Parameters
            (m, d, a, alpha) = ur_parameter_function(zeros_array)

            P = get_P(a, d, alpha, vector, cos, sin)

            PC = ur_frames_fun(a, vector)

            if inertia_is_wrt_CoM:
                I_CoM = inertia
            else:
                I_CoM = [zeros_matrix(3, 3) for i in range(0, Njoints + 1)]
                for i in range(1, Njoints + 1):
                    PC_dot_left = dot(PC[i].transpose(), PC[i])
                    PC_dot_right = dot(PC[i], (PC[i].transpose()))
                    assert PC_dot_left.shape == (1, 1)
                    assert PC_dot_right.shape == (3, 3)
                    PC_dot_scalar = PC_dot_left[0, 0]
                    I_CoM[i] = inertia[i] - m[i] * (PC_dot_scalar * identity_3 - PC_dot_right)

            # State
            w = [vector([0, 0, 0]) for i in range(0, Njoints + 1)]  # Angular velocity
            wd = [vector([0, 0, 0]) for i in range(0, Njoints + 1)]  # Angular acceleration
            vd = [vector([0, 0, 0]) for i in range(0, Njoints + 1)]  # Translational acceleration

            # Gravity
            vd[0] = -g

            vcd = [vector([0, 0, 0]) for i in range(0, Njoints + 1)]
            F = [vector([0, 0, 0]) for i in range(0, Njoints + 1)]
            N = [vector([0, 0, 0]) for i in range(0, Njoints + 1)]

            f = [vector([0, 0, 0]) for i in range(0, Njoints + 1)]
            f.append(f_tcp)

            n = [vector([0, 0, 0]) for i in range(0, Njoints + 1)]
            n.append(n_tcp)

            Z = vector([0, 0, 1])
            (R_i_im1, R_im1_i, _) = get_forward_kinematics(q, alpha, P, zeros_matrix, matrix, cos, sin)

            # Outputs
            tau = [zeros_matrix(1, 1) for i in range(0, Njoints + 1)]

            # Outward calculations i: 0 -> 5
            for i in range(0, Njoints):
                w[i+1] = dot(R_im1_i[i+1], w[i]) + qd[i+1] * Z
                assert w[i+1].shape == (3, 1)
                wd[i+1] = dot(R_im1_i[i+1], wd[i]) + cross(dot(R_im1_i[i+1], w[i]), qd[i+1] * Z) + qdd[i+1] * Z
                assert wd[i+1].shape == (3, 1)
                assert vd[i].shape == (3, 1)
                vd[i+1] = dot(R_im1_i[i+1], cross(wd[i], P[i+1]) + cross(w[i], cross(w[i], P[i+1])) + vd[i])
                assert vd[i+1].shape == (3, 1)
                vcd[i+1] = cross(wd[i+1], PC[i+1]) + cross(w[i+1], cross(w[i+1], PC[i+1])) + vd[i+1]
                assert vcd[i+1].shape == (3, 1)
                F[i+1] = m[i+1] * vcd[i+1]
                assert F[i+1].shape == (3, 1)
                N[i+1] = dot(I_CoM[i+1], wd[i+1]) + cross(w[i+1], dot(I_CoM[i+1], w[i+1]))
                assert N[i+1].shape == (3, 1)

            # Inward calculations i: 6 -> 1
            for j in range(0, Njoints):
                i = Njoints - j
                f[i] = dot(R_i_im1[i+1], f[i+1]) + F[i]
                n[i] = N[i] + dot(R_i_im1[i+1], n[i+1]) + cross(PC[i], F[i]) + cross(P[i+1], dot(R_i_im1[i+1], f[i+1]))
                assert n[i].shape == (3, 1), n[i]
                tau[i] = dot(n[i].transpose(), Z)
                assert tau[i].shape == (1, 1)

            tau_list = [t[0, 0] for t in tau]

            return tau_list

        return compute_torques

    return create_torque_algorithm


create_torque_algorithm_3e = torque_algorithm_factory_from_parameters(get_ur3e_parameters, get_ur3e_PC)

compute_torques_numeric_3e = create_torque_algorithm_3e(npvector, npmatrix,
                                                        npzeros_array, npzeros_matrix,
                                                        npcos, npsin,
                                                        npcross, npdot,
                                                        npeye)

compute_torques_symbolic_3e = create_torque_algorithm_3e(spvector, spmatrix,
                                                         spzeros_array, spzeros_matrix,
                                                         spcos, spsin,
                                                         spcross, spdot,
                                                         speye)


create_torque_algorithm_5e = torque_algorithm_factory_from_parameters(get_ur5e_parameters, get_ur5e_PC)

compute_torques_numeric_5e = create_torque_algorithm_5e(npvector, npmatrix,
                                                        npzeros_array, npzeros_matrix,
                                                        npcos, npsin,
                                                        npcross, npdot,
                                                        npeye)

compute_torques_symbolic_5e = create_torque_algorithm_5e(spvector, spmatrix,
                                                         spzeros_array, spzeros_matrix,
                                                         spcos, spsin,
                                                         spcross, spdot,
                                                         speye)

create_torque_algorithm_ur = torque_algorithm_factory_from_parameters(get_ur_parameters_symbolic, get_ur_frames)
compute_torques_symbolic_ur = create_torque_algorithm_ur(spvector, spmatrix,
                                                         spzeros_array, spzeros_matrix,
                                                         spcos, spsin,
                                                         spcross, spdot,
                                                         speye)
