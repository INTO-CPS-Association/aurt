from aurt.globals import Njoints


def get_forward_kinematics(q, alpha, P, zeros_matrix, matrix, cos, sin):
    c = lambda i: cos(q[i])
    s = lambda i: sin(q[i])

    R_i_im1 = [zeros_matrix(3, 3) for i in range(0, Njoints + 2)]
    for i in range(1, Njoints + 1):
        # i=1,...,6
        R_i_im1[i] = matrix([
            [c(i),                    -s(i),                   0],
            [s(i) * cos(alpha[i-1]),  c(i) * cos(alpha[i-1]),  -sin(alpha[i-1])],
            [s(i) * sin(alpha[i-1]),  c(i) * sin(alpha[i-1]),  cos(alpha[i-1])]
        ])

    # R_i_im1[1] means the orientation of frame 1 wrt to 0.
    # R_i_im1[4] means the orientation of frame 4 wrt to 3.
    # R_i_im1[4+1] = orientation of frame 5 to frame 4

    R_im1_i = [r.transpose() for r in R_i_im1]
    # R_im1_i[4] means the rotation from frame 3 to 4.

    # Construct matrix T_i_im1.
    # T_i_im1[i] means the transformation from a point in frame {i} to frame {i-1}
    # T_i_im1[4] means the transformation from a point in frame {4} to frame {3}
    T_i_im1 = [zeros_matrix(4, 4) for i in range(0, Njoints + 2)]
    for i in range(1, Njoints + 1):
        # i=1,...,6
        T_i_im1[i] = matrix([
            [R_i_im1[i][0, 0], R_i_im1[i][0, 1], R_i_im1[i][0, 2], P[i][0,0]],
            [R_i_im1[i][1, 0], R_i_im1[i][1, 1], R_i_im1[i][1, 2], P[i][1,0]],
            [R_i_im1[i][2, 0], R_i_im1[i][2, 1], R_i_im1[i][2, 2], P[i][2,0]],
            [0, 0, 0, 1]
        ])

    return R_i_im1, R_im1_i, T_i_im1
