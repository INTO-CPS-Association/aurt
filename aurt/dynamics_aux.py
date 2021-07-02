import sys

import sympy as sp
from itertools import chain
import time  # TODO: Delete


def sym_mat_to_subs(sym_mats, num_mats):
    subs = {}

    for s_mat, n_mat in zip(sym_mats, num_mats):
        subs = {**subs, **{s: v for s, v in zip(s_mat, n_mat) if s != 0}}

    return subs


def replace_first_moments(args):

    tau_sym_j = args[0][0]  # tau_sym[j]
    j = args[0][1]
    m = args[1]  # [m1, ..., mN]
    pc = args[2]  # PC = [PC[1], ..., PC[N]], PC[j] = [PCxj, PCyj, PCzj]
    m_pc = args[3]  # mPC = [mPC[1], ..., mPC[N]], mPC[j] = [mXj, mYj, mZj]
    n_joints = args[4]

    # TODO: REMOVE THE 'IGNORE J=0' STUFF
    if j == 0:
        print(f"Ignore joint {j}...")
        return tau_sym_j

    n_cartesian = len(m_pc[0])
    for jj in range(j, n_joints + 1):  # The joint j torque equation is affected by dynamic coefficients only for links jj >= j
        for i in range(n_cartesian):  # Cartesian coordinates loop
            # Print progression counter
            total = (n_joints + 1 - j) * n_cartesian
            progress = (jj - j) * n_cartesian + i + 1
            print(f"Task {j}: {progress}/{total} (tau[{j}]: {m[jj]}*{pc[jj][i]} -> {m_pc[jj][i]})")
            tau_sym_j = tau_sym_j.expand().subs(m[jj] * pc[jj][i], m_pc[jj][i])  # expand() is - for unknown reasons - needed for subs() to consistently detect the "m*PC" products

    return tau_sym_j


def compute_regressor_row(args):  # TODO: remove 'reg_j_new'
    """
    We compute the regressor via symbolic differentiation. Each torque equation must be linearizable with respect to
    the dynamic coefficients.

    Example:
        Given the equation 'tau = sin(q)*a', tau is linearizable with respect to 'a', and the regressor 'sin(q)' can be
        obtained by partial differentiation of 'tau' with respect to 'a'.
    """
    tau_sym_linearizable_j = args[0][0]  # tau_sym_linearizable[j]
    j = args[0][1]
    n_joints = args[1]
    p_linear = args[2]

    n_par = [len(p_linear[j]) for j in range(len(p_linear))]
    reg_row_j = sp.zeros(1, sum(n_par))  # Initialization
    # reg_row_j_new = sp.zeros(1, sum(n_par))  # Initialization

    if j == 0:
        print(f"Ignore joint {j}...")
        return reg_row_j

    # For joint j, we loop through the parameters belonging to joints/links >= j. This is because it is physically
    # impossible for torque eq. j to include a parameter related to proximal links (< j). We divide the parameter
    # loop (for joints >= j) in two variables 'jj' and 'i':
    #   'jj' describes the joint, which the parameter belongs to
    #   'i'  describes the parameter's index/number for joint jj.
    t00 = time.time()
    for jj in range(j, n_joints + 1):  # Joint loop including this and distal (succeeding) joints (jj >= j)
        for i in range(n_par[jj]):  # joint jj parameter loop
            column_idx = sum(n_par[:jj]) + i
            print(
                f"Computing regressor(row={j}/{n_joints + 1}, column={column_idx + 1}/{sum(n_par)}) by analyzing dependency of tau[{j}] on joint {jj}'s parameter {i}: {p_linear[jj][i]}")
            reg_row_j[0, column_idx] = sp.diff(tau_sym_linearizable_j, p_linear[jj][i])
    t01 = time.time()
    print(f"Time original: {t01 - t00} s")

    # p_linear_global = list(chain.from_iterable(p_linear))
    # sys.setrecursionlimit(int(1e6))  # Prevents errors in sympy lambdify
    # t10 = time.time()
    # sym_par_j = list(chain.from_iterable(p_linear[j:n_joints+1][:]))
    # standard_basis_j = sp.eye(len(sym_par_j)).tolist()
    # tau_sym_linearizable_j_func = sp.lambdify(sym_par_j, tau_sym_linearizable_j, "sympy")
    # for jj in range(j, n_joints + 1):  # Joint loop including this and distal (succeeding) joints (jj >= j)
    #     for i in range(n_par[jj]):  # joint jj parameter loop
    #         column_idx = sum(n_par[:jj]) + i
    #         # print(
    #         #     f"Computing regressor(row={j}/{n_joints + 1}, column={column_idx + 1}/{sum(n_par)}) by analyzing dependency of tau[{j}] on joint {jj}'s parameter {i}: {p_linear[jj][i]}")
    #         # reg_j_new[0, column_idx] = tau_sym_linearizable_j.subs(zip(p_linear_global, standard_basis[column_idx]))  # THIS IS MUCH SLOWER
    #         # reg_row_j[0, column_idx] = tau_sym_linearizable_j_func(*standard_basis[column_idx])
    #         reg_row_j[0, column_idx] = tau_sym_linearizable_j_func(*standard_basis[column_idx])
    # t11 = time.time()
    # print(f"Time new: {t11 - t10} s")
    # print(f"*** Joint {j} *** Time original: {t01 - t00} s,  Time new: {t11 - t10} s, new is faster? {t11 - t10 < t01 - t00}")
    # print(f"Relative speed-up: {(t01 - t00-(t11 - t10))/(t11 - t10)*100} %")

    # assert reg_row_j_new == reg_row_j, "results are not equal!"
    # print(f"Assertion validated, delete computationally slowest (reg_j?) method of compute_regressor_row in dynamics.aux...")

    return reg_row_j
