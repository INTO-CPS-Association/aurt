import os
from itertools import product, chain

import sys
import unittest
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import sympy as sp
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from aurt.calibration_aux import find_nonstatic_start_and_end_indices
from aurt.robot_data import RobotData, plot_colors
from aurt.file_system import cache_object, store_object, load_object, project_root
from aurt.globals import Njoints, get_ur_parameters_symbolic, get_ur_frames, get_ur5e_parameters
from aurt.num_sym_layers import spzeros_array, spvector, npzeros_array
from aurt.torques import compute_torques_symbolic_ur
from aurt.robot_dynamics import RobotDynamics
from aurt.robot_calibration import RobotCalibration
from aurt.data_processing import convert_file_to_mdh
from aurt.joint_dynamics import JointDynamics
from tests import NONINTERACTIVE
from tests.utils.timed_test import TimedTest

q = [0.0] + [sp.symbols(f"q{j}") for j in range(1, Njoints + 1)]
qd = [0.0] + [sp.symbols(f"qd{j}") for j in range(1, Njoints + 1)]
qdd = [0.0] + [sp.symbols(f"qdd{j}") for j in range(1, Njoints + 1)]

# Rigid-body dynamics
mX = sp.symbols([f"mX{j}" for j in range(Njoints + 1)])
mY = sp.symbols([f"mY{j}" for j in range(Njoints + 1)])
mZ = sp.symbols([f"mZ{j}" for j in range(Njoints + 1)])
XX = sp.symbols([f"XX{j}" for j in range(Njoints + 1)])
XY = sp.symbols([f"XY{j}" for j in range(Njoints + 1)])
XZ = sp.symbols([f"XZ{j}" for j in range(Njoints + 1)])
YY = sp.symbols([f"YY{j}" for j in range(Njoints + 1)])
YZ = sp.symbols([f"YZ{j}" for j in range(Njoints + 1)])
ZZ = sp.symbols([f"ZZ{j}" for j in range(Njoints + 1)])
i_cor = [sp.zeros(3, 3) for i in range(Njoints + 1)]
for j in range(Njoints + 1):
    i_cor[j] = sp.Matrix([
        [XX[j], XY[j], XZ[j]],
        [XY[j], YY[j], YZ[j]],
        [XZ[j], YZ[j], ZZ[j]]
    ])
gx, gy, gz = sp.symbols(f"gx gy gz")
g = sp.Matrix([gx, gy, gz])
g_num = np.array([0.0, 0.0, 9.81])

# Force & torque at the tool center point (TCP)
fx, fy, fz, nx, ny, nz = sp.symbols(f"fx fy fz nx ny nz")
f_tcp = sp.Matrix([fx, fy, fz])  # Force at the TCP
n_tcp = sp.Matrix([nx, ny, nz])  # Moment at the TCP
f_tcp_num = sp.Matrix([0.0, 0.0, 0.0])
n_tcp_num = sp.Matrix([0.0, 0.0, 0.0])

# PAYLOAD PARAMETERS
XX_pl, XY_pl, XZ_pl, YY_pl, YZ_pl, ZZ_pl = sp.symbols(f"XX_pl XY_pl XZ_pl YY_pl YZ_pl ZZ_pl")  # payload inertia
mX_pl, mY_pl, mZ_pl = sp.symbols(f"mX_pl mY_pl mZ_pl")
m_pl = sp.symbols(f"m_pl")

# Joint dynamics
load_model = 'square'  # 'none', 'square' or 'abs'
hysteresis_model = 'sgn'  # 'sgn' or 'gms'
viscous_friction_powers = [1]  # [1, 2, 3, ...], i.e. list of positive integers

Fc = sp.symbols([f"Fc{j}" for j in range(Njoints + 1)])
Fvs = [[sp.symbols(f"Fv{j}_{i}") for i in viscous_friction_powers] for j in range(Njoints + 1)]
Fcl = sp.symbols([f"Fcl{j}" for j in range(Njoints + 1)])
tauJ = sp.symbols([f"tauJ{j}" for j in range(Njoints + 1)])

# *************************************************** NOT IN USE ATM ***************************************************
# GMS parameters
max_gms_elements = 10
gms_spacing_power_law = 4.0
dq = sp.symbols([f"dq{j}" for j in range(Njoints)])  # [rad] Change of angular position of joint j (since previous sample)
n_gms_elements = 4
z0j = [sp.Matrix([[sp.Symbol(f"z0{j}_{i}") for i in range(n_gms_elements)]]) for j in range(Njoints)]
# z1j = sp.Matrix(sp.zeros(n_gms_elements, 1))
k_gms = [sp.Matrix([[sp.Symbol(f"k{j}_{i}") for i in range(n_gms_elements)]]) for j in range(Njoints)]
backlash = [sp.Symbol(f"backlash{j}") for j in range(Njoints)]
# **********************************************************************************************************************

f_dyn = 10.0  # Approximate frequency of the robot dynamics

(m, d, a, alpha) = get_ur_parameters_symbolic(spzeros_array)
PC = get_ur_frames(None, spvector)
mPC = [[mX[j], mY[j], mZ[j]] for j in range(Njoints + 1)]
p_joint = [[Fc[j], *Fvs[j]] for j in range(Njoints + 1)]  # Joint dynamics parameters
p_physical = [[XX[j], XY[j], XZ[j], YY[j], YZ[j], ZZ[j], PC[j][0], PC[j][1], PC[j][2], m[j]] for j in range(Njoints + 1)]  # Physical parameters
p_linear = [[XX[j], XY[j], XZ[j], YY[j], YZ[j], ZZ[j], mX[j], mY[j], mZ[j], m[j]] + p_joint[j] for j in range(Njoints + 1)]  # Linear parameters
n_par_j_joint_dynamics = len(p_joint[0])
n_par_j_rbd_revolute = len(p_physical[0])
n_par_linear = [n_par_j_rbd_revolute + n_par_j_joint_dynamics]*(Njoints + 1)
# Initialize in global scope (very bad coding standard?)
p_linear_exist = []*(Njoints+1)
p_base = []*(Njoints+1)
idx_linear_exist = []*(Njoints+1)
idx_base = []*(Njoints+1)
n_par_linear_exist = []*(Njoints+1)
n_par_base = [0]*(Njoints+1)

# ************************ These functions should all be changed to take the argument 'joint' **************************


def coulomb_friction_parameters():
    """
    Returns a list of 'Njoints + 1' elements, each element comprising a list of Coulomb friction parameters for each
    joint.
    """

    assert load_model.lower() in ('none', 'square', 'abs')

    if load_model.lower() == 'none':
        return [[Fc[j]] for j in range(Njoints + 1)]
    else:
        return [[Fc[j], Fcl[j]] for j in range(Njoints + 1)]


def viscous_friction_parameters():
    """
    Returns a list of 'Njoints + 1' elements, each element comprising a list of viscous friction parameters for each
    joint.
    """

    return [[sp.symbols(f"Fv{j}_{i}") for i in viscous_friction_powers] for j in range(Njoints + 1)]


def joint_dynamics_parameters():
    """
    Returns a list of 'Njoints + 1' elements with each element comprising a list of joint dynamics parameters for that
    corresponding joint.
    """

    Fcs = coulomb_friction_parameters()
    Fvs = viscous_friction_parameters()
    return [[*Fcs[j], *Fvs[j]] for j in range(Njoints + 1)]


def rigid_body_dynamics_parameters():
    """
    Returns a list of 'Njoints + 1' elements with each element compising a list of all rigid body parameters related to
    that corresponding link.
    """
    return [[XX[j], XY[j], XZ[j], YY[j], YZ[j], ZZ[j], mX[j], mY[j], mZ[j], m[j]] for j in range(Njoints + 1)]


def linear_parameters():
    """
    Returns a list of 'Njoints + 1' elements with each element comprising a list of all parameters related to that
    corresponding link and joint.
    """
    par_joint = joint_dynamics_parameters()
    par_rigid_body = rigid_body_dynamics_parameters()
    return [par_rigid_body[j] + par_joint[j] for j in range(Njoints + 1)]


def coulomb_friction_basis():
    """
    Returns the basis (regressor) of the Coulomb friction model.
    """

    assert load_model.lower() in ('none', 'square', 'abs')

    h = generalized_hysteresis_model()
    if load_model.lower() == 'none':
        return [[h[j]] for j in range(Njoints + 1)]
    elif load_model.lower() == 'square':
        return [[h[j], h[j]*tauJ[j]**2] for j in range(Njoints + 1)]
    else:  # load_model == 'abs'
        return [[h[j], h[j]*abs(tauJ[j])] for j in range(Njoints + 1)]


def generalized_hysteresis_model():
    """
    Returns the hysteresis model used in the Coulomb friction model.
    """
    assert hysteresis_model.lower() in ('sgn', 'gms')

    if hysteresis_model.lower() == 'sgn':
        return [sp.sign(qd[j]) for j in range(Njoints + 1)]
    else:
        print(f"GMS model not yet implemented - using a simple static model instead.")
        return [sp.sign(qd[j]) for j in range(Njoints + 1)]


def viscous_friction_basis():
    """
    Returns the basis (regressor) of the viscous friction model.
    """

    fv_basis = [[sp.zeros(1, 1)]*len(viscous_friction_powers) for _ in range(7)]
    for j in range(Njoints + 1):
        for i, power in enumerate(viscous_friction_powers):
            if power % 2 == 0:  # even
                fv_basis[j][i] = qd[j] ** (power - 1) * abs(qd[j])
            else:  # odd
                fv_basis[j][i] = qd[j] ** power
    return fv_basis


def joint_dynamics_basis():
    """
    Returns a list of 'Njoints + 1' elements with each element comprising a list of joint dynamics parameters for that
    corresponding joint.
    """

    fc_basis = coulomb_friction_basis()
    fv_basis = viscous_friction_basis()
    return [[*fc_basis[j], *fv_basis[j]] for j in range(Njoints + 1)]


def joint_dynamics():
    """
    Returns a list of joint dynamics with the list elements corresponding to the joint dynamics of each joint.
    """
    jd_basis = joint_dynamics_basis()
    jd_par = joint_dynamics_parameters()
    return [sum([b*p for b, p in zip(jd_basis[j], jd_par[j])]) for j in range(len(jd_basis))]

# **********************************************************************************************************************


def number_of_parameters_joint_dynamics():
    return len(joint_dynamics_parameters()[0])


def number_of_parameters_rigid_body_dynamics():
    return len(rigid_body_dynamics_parameters()[0])


def number_of_parameters_linear_dynamics():
    n_par_linear = [number_of_parameters_rigid_body_dynamics() + number_of_parameters_joint_dynamics()] * (Njoints + 1)
    return n_par_linear


def normalized_root_mean_squared_error_as_fit_percentage(y_true, y_pred):
    """This function calculates the normalized root mean squared error of each channel in y. The result is expressed in
    percentage fitness, where a fitness of 0 % corresponds to a model equal to the mean value of the data and a fitness
    of 100 % corresponds to a model that fits perfectly."""
    assert y_true.shape == y_pred.shape

    n_channels = y_true.shape[0]  # no. of channels
    mean = np.mean(y_true, axis=1)
    nrmse_fit = np.zeros(n_channels)
    for i in range(n_channels):
        nrmse_fit[i] = 1 - np.linalg.norm(y_true[i, :] - y_pred[i, :]) / np.linalg.norm(y_true[i, :] - mean[i])
    return nrmse_fit


def get_mse(y_meas, y_pred):
    """This function calculates the mean squared error of each channel in y using sklearn.metrics.mean_squared_error."""
    assert y_meas.shape == y_pred.shape

    n_channels = y_meas.shape[0]  # no. of channels
    mse = np.zeros(n_channels)
    for i in range(n_channels):
        mse[i] = mean_squared_error(y_meas[i, :], y_pred[i, :])
    return mse


def sym_mat_to_subs(sym_mats, num_mats):
    subs = {}

    for s_mat, n_mat in zip(sym_mats, num_mats):
        subs = {**subs, **{s: v for s, v in zip(s_mat, n_mat) if s != 0}}

    return subs


def replace_first_moments(args):

    tau_sym_j = args[0][0]  # tau_sym[j]
    j = args[0][1]
    m = args[1]  # [m1, ..., mN]
    PC = args[2]  # PC = [PC[1], ..., PC[N]], PC[j] = [PCxj, PCyj, PCzj]
    m_pc = args[3]  # mPC = [mPC[1], ..., mPC[N]], mPC[j] = [mXj, mYj, mZj]

    if j == 0:
        print(f"Ignore joint {j}...")
        return tau_sym_j

    n_cartesian = len(m_pc[0])
    for jj in range(j, Njoints+1):  # The joint j torque equation is affected by dynamic coefficients only for links jj >= j
        for i in range(n_cartesian):  # Cartesian coordinates loop
            # Print progression counter
            total = (Njoints+1-j)*n_cartesian
            progress = (jj - j)*n_cartesian + i + 1
            print(f"Task {j}: {progress}/{total} (tau[{j}]: {m[jj]}*{PC[jj][i]} -> {m_pc[jj][i]})")
            tau_sym_j = tau_sym_j.expand().subs(m[jj] * PC[jj][i], m_pc[jj][i])  # expand() is - for unknown reasons - needed for subs() to consistently detect the "m*PC" products

    return tau_sym_j


def compute_regressor_row(args):
    """We compute the regressor via symbolic differentiation. Each torque equation must be linearizable with respect to
    the dynamic coefficients.

    Example:
        Given the equation 'tau = sin(q)*a', tau is linearizable with respect to 'a', and the regressor 'sin(q)' can be
        obtained by partial differentiation of 'tau' with respect to 'a'.
    """
    tau_sym_linearizable_j = args[0][0]  # tau_sym_linearizable[j]
    j = args[0][1]
    p_linear = args[1]
    reg_j = sp.zeros(1, sum(n_par_linear))  # Initialization

    if j == 0:
        print(f"Ignore joint {j}...")
        return reg_j

    # For joint j, we loop through the parameters belonging to joints/links >= j. This is because it is physically
    # impossible for torque eq. j to include a parameter related to proximal links (< j). We divide the parameter
    # loop (for joints >= j) in two variables 'jj' and 'i':
    #   'jj' describes the joint, which the parameter belongs to
    #   'i'  describes the parameter's index/number for joint jj.
    for jj in range(j, Njoints + 1):  # Joint loop including this and distal (succeeding) joints (jj >= j)
        for i in range(n_par_linear[jj]):  # joint jj parameter loop
            column_idx = sum(n_par_linear[:jj]) + i
            print(f"Computing regressor(row={j}/{Njoints + 1}, column={column_idx}/{sum(n_par_linear)}) by analyzing dependency of tau[{j}] on joint {jj}'s parameter {i}: {p_linear[jj][i]}")
            reg_j[0, column_idx] = sp.diff(tau_sym_linearizable_j, p_linear[jj][i])

    return reg_j


def compute_tau_sym_linearizable():
    # TODO: The cache files should include information about the robot. Also, ideally, it should be changed so that the
    #  joint dynamics are added to the regressor later.

    tau_sym_rbd_linearizable = cache_object('./rigid_body_dynamics_linearizable', compute_tau_rbd_linearizable_parallel)

    tau_sym_jd_linearizable = cache_object(
        f'./joint_dynamics_linearizable_{load_model}_{hysteresis_model}_{viscous_friction_powers}', joint_dynamics)

    return sp.Matrix([tau_sym_rbd_linearizable[j] + tau_sym_jd_linearizable[j] for j in range(len(tau_sym_rbd_linearizable))])


def compute_tau_rbd_linearizable_parallel():
    tau_sym_rbd = cache_object('./rigid_body_dynamics',
                               lambda: compute_torques_symbolic_ur(q, qd, qdd, f_tcp, n_tcp, i_cor, g,
                                                                   inertia_is_wrt_CoM=False))

    js = list(range(Njoints + 1))
    tau_per_task = [tau_sym_rbd[j] for j in js]  # Allows one to control how many tasks by controlling how many js.
    data_per_task = list(product(zip(tau_per_task, js), [m], [PC], [mPC]))

    with Pool() as p:
        tau_sym_rbd_linearizable = p.map(replace_first_moments, data_per_task)

    return tau_sym_rbd_linearizable

"""
# ************************************************** NOT IN USE ATM ****************************************************
def gms_friction(j):
    # NOT VALIDATED
    assert 0 < n_gms_elements < max_gms_elements

    # Deltaj = [np.power(float(i)/n_elements, gms_spacing_power_law) for i in range(n_elements)]
    #
    # for i in range(n_elements):
    #     z1[j][i] = sp.sign(z0j[i] + dq[j]) * min(abs(z0j[i] + dq[j]), backlash[j]*Deltaj[i])  # State equation
    #
    # yj = k_gms[j].T * z1[j] / backlash[j]  # Friction torque
    #
    # return yj


def compute_gms_states(dqj, z0j, backlashj):
    # NOT VALIDATED
    n_samples = dqj.size
    z1j = np.zeros_like(z0j)
    n_elements = z1j.shape[1]
    spacing_power_law = 4.0
    Deltaj = [np.power(float(i) / n_elements, spacing_power_law) for i in range(n_elements)]

    for i in range(n_elements):
        z1j[i] = np.sign(z0j[i] + dqj) * np.min(np.abs(z0j[i] + dqj), backlashj*Deltaj[i])  # State equation

    return z1j
# **********************************************************************************************************************
"""


def check_symbolic_linear_system(tau, regressor_matrix, parameter_vector, joints=None):
    """Symbolically checks that the regressor matrix times the parameter vector equals tau"""
    if joints is None:
        joints = list(range(1, Njoints+1))

    for j in joints:
        reg_lin_mul_j = regressor_matrix[j, :].dot(parameter_vector)
        assert sp.simplify(tau[j] - reg_lin_mul_j) == 0, f"Joint {j}: FAIL!"


def compute_regressor_parallel():
    tau_sym_linearizable = cache_object('./dynamics_linearizable', compute_tau_sym_linearizable)

    js = list(range(Njoints + 1))
    tau_per_task = [tau_sym_linearizable[j] for j in js]  # Allows one to control how many tasks by controlling how many js
    data_per_task = list(product(zip(tau_per_task, js), [linear_parameters()]))

    with Pool() as p:
        reg = p.map(compute_regressor_row, data_per_task)

    return sp.Matrix(reg)


def compute_regressor_linear_exist(regressor=None):
    # ************************* IDENTIFY (LINEAR) PARAMETERS WITH NO EFFECT ON THE DYNAMICS ************************
    # In the eq. for joint j, the dynamic parameters of proximal links (joints < j, i.e. closer to the base) will never
    # exist, i.e. the dynamic parameter of joint j will not be part of the equations for joints > j.
    if regressor is None:
        regressor = cache_object('./regressor', compute_regressor_parallel)

    filename = 'parameter_indices_linear_exist'
    idx_linear_exist = cache_object(filename, lambda: compute_parameters_linear_exist(regressor)[0])

    # Removes zero columns of regressor corresponding to parameters with no influence on dynamics
    idx_linear_exist_global = np.where(list(chain.from_iterable(idx_linear_exist)))[0].tolist()
    return regressor[:, idx_linear_exist_global]


def compute_parameters_linear_exist(regressor=None):
    # In the regressor, identify the zero columns corresponding to parameters which does not matter to the system.
    filename_idx = 'parameter_indices_linear_exist'
    filename_npar = 'number_of_parameters_linear_exist'
    filename_par = 'parameters_linear_exist'

    if not (os.path.isfile(filename_idx) and os.path.isfile(filename_npar) and os.path.isfile(filename_par)):
        if regressor is None:
            regressor = cache_object('./regressor', compute_regressor_parallel)

        idx_linear_exist = [[True for _ in range(n_par_linear[j])] for j in range(len(n_par_linear))]
        n_par_linear_exist = n_par_linear.copy()
        p_linear_exist = p_linear.copy()
        print(f"p_linear: {p_linear}")
        for j in range(Njoints+1):
            for i in reversed(range(n_par_linear[j])):  # parameter loop
                idx_column = sum(n_par_linear[:j]) + i
                if not any(regressor[:, idx_column]):
                    idx_linear_exist[j][i] = False
                    n_par_linear_exist[j] -= 1
                    del p_linear_exist[j][i]

        store_object(idx_linear_exist, filename_idx)
        store_object(n_par_linear_exist, filename_npar)
        store_object(p_linear_exist, filename_par)
    else:
        idx_linear_exist = load_object(filename_idx)
        n_par_linear_exist = load_object(filename_npar)
        p_linear_exist = load_object(filename_par)

    return idx_linear_exist, n_par_linear_exist, p_linear_exist


def compute_parameters_base():
    """Identifies the indices of the """
    # ********************************* IDENTIFY BASE PARAMETERS OF THE DYNAMICS SYSTEM ********************************
    sys.setrecursionlimit(int(1e6))  # Prevents errors in sympy lambdify
    if load_model.lower() == 'none':
        args_sym = q[1:] + qd[1:] + qdd[1:]
    else:
        args_sym = q[1:] + qd[1:] + qdd[1:] + tauJ[1:]

    regressor_with_instantiated_parameters_func = sp.lambdify(args_sym, cache_object('./regressor_with_instantiated_parameters',
                                                         lambda: compute_regressor_with_instantiated_parameters(
                                                             ur_param_function=get_ur5e_parameters)))
    filename = 'parameter_indices_base'
    idx_base_global = cache_object(filename, lambda: compute_indices_base_exist(regressor_with_instantiated_parameters_func))

    idx_linear_exist, n_par_linear_exist, p_linear_exist = compute_parameters_linear_exist()
    p_linear_exist_vector = sp.Matrix(list(chain.from_iterable(p_linear_exist))) # flatten 2D list to 1D list and convert 1D list to sympy.Matrix object

    filename = 'parameters_base_exist'
    p_base_vector = cache_object(filename, lambda: p_linear_exist_vector[idx_base_global, :])

    # Initialization
    n_par_base = n_par_linear_exist.copy()
    p_base = p_linear_exist.copy()
    idx_is_base = [[True for _ in range(len(p_base[j]))] for j in range(len(p_base))]

    print(f"p_base: {p_base}")
    for j in range(Njoints+1):
        for i in reversed(range(n_par_linear_exist[j])):
            if not p_base[j][i] in p_base_vector.free_symbols:
                idx_is_base[j][i] = False
                n_par_base[j] -= 1
                del p_base[j][i]
    print(f"p_base: {p_base}")

    assert np.count_nonzero(list(chain.from_iterable(idx_is_base))) == sum(n_par_base) == len(list(chain.from_iterable(p_base)))

    return idx_is_base, n_par_base, p_base


def trajectory_filtering_and_central_difference(q_m, dt, idx_start, idx_end):
    trajectory_filter_order = 4
    cutoff_freq_trajectory = 5 * f_dyn  # Cut-off frequency should be around 5*f_dyn = 50 Hz(?)
    trajectory_filter = signal.butter(trajectory_filter_order, cutoff_freq_trajectory, btype='low', output='sos', fs=1 / dt)
    q_tf = signal.sosfiltfilt(trajectory_filter, q_m, axis=1)

    # Obtain first and seond order time-derivatives of measured and filtered trajectory
    qd_tf = np.gradient(q_tf, dt, edge_order=2, axis=1)
    # Using the gradient function a second time to obtain the second-order time derivative would result in
    # additional unwanted smoothing, see https://stackoverflow.com/questions/23419193/second-order-gradient-in-numpy
    qdd_tf = (q_tf[:, 2:] - 2 * q_tf[:, 1:-1] + q_tf[:, :-2]) / (dt ** 2)  # two fewer indices than q and qd

    # Truncate data
    q_tf = q_tf[:, idx_start:idx_end]
    qd_tf = qd_tf[:, idx_start:idx_end]
    qdd_tf = qdd_tf[:, idx_start-1:idx_end-1]  # shifted due to a "lost" index in the start of the dataset

    assert q_tf.shape == qd_tf.shape == qdd_tf.shape

    return q_tf, qd_tf, qdd_tf


def parallel_filter(y, dt):
    """Applies a 4th order Butterworth (IIR) filter for each row in y having a cutoff frequency of 2*f_dyn."""
    # Link to cut-off freq. eq.: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6151858
    parallel_filter_order = 4
    cutoff_freq_parallel = 2 * f_dyn
    parallel_filter = signal.butter(parallel_filter_order, cutoff_freq_parallel, btype='low', output='sos', fs=1/dt)

    y_pf = signal.sosfiltfilt(parallel_filter, y, axis=0)
    return y_pf


def downsample(y, dt):
    """The decimate procedure down-samples the signal such that the matrix system (that is later to be inverted) is not
    larger than strictly required. The signal.decimate() function can also low-pass filter the signal before
    down-sampling, but for IIR filters unfortunately only the Chebyshev filter is available which has (unwanted) ripple
    in the passband unlike the Butterworth filter that we use. The approach for downsampling is simply picking every
    downsampling_factor'th sample of the data."""

    downsampling_factor = round(0.8 / (4*f_dyn*dt))  # downsampling_factor = 10 for dt = 0.002 s and f_dyn = 10 Hz
    y_ds = y[::downsampling_factor, :]
    return y_ds


def compute_joint_torque_basis(q_num, qd_num, qdd_num, robot_param_function):
    """
    This method computes the basis of the joint torque by evaluating the rigid body dynamics with instantiated
    DH parameters, gravity, and TCP force/torque. All dynamic parameters are set equal to ones. Thus, the output of this
    method is not strictly equal to the joint torque.
    """
    assert q_num.shape == qd_num.shape == qdd_num.shape

    sys.setrecursionlimit(int(1e6))  # Prevents errors in sympy lambdify

    # 1. load (Njoints+1) list of rigid-body dynamics
    # 2. instantiate parameters (DH, gravity, TCP force/torque)
    # 3. instantiate ones for parameters
    # 4. lambdify in terms of q, qd, and qdd
    # 5. evaluate in q_num, qd_num, and qdd_num
    tauJ_basis = compute_joint_torque_basis_with_instantiated_parameters(robot_param_function)
    args_sym = q[1:] + qd[1:] + qdd[1:]
    args_num = np.concatenate((q_num, qd_num, qdd_num))
    tauJ_num = np.zeros((tauJ_basis.shape[0], args_num.shape[1]))
    for j in range(Njoints+1):
        tauJ_fcn_j = sp.lambdify(args_sym, tauJ_basis[j])

        tauJ_num[j, :] = tauJ_fcn_j(*args_num)
    return tauJ_num


def compute_joint_torque_basis_with_instantiated_parameters(robot_param_function):
    """
    This function computes the joint torque basis as the rigid-body dynamics with parameters (DH, gravity, and TCP
    force/torque) instantiated. The parameters related to the rigid-body dynamics (masses, center-of-mass positions, and
    inertia components) are set to one.
    """

    (_, d_num, a_num, _) = robot_param_function(npzeros_array)

    def to_fname(l):
        return "_".join(map(lambda s: "%1.2f" % s, l))

    data_id = f"{to_fname(d_num)}_{to_fname(a_num)}_{'%1.2f' % g_num[0]}_{'%1.2f' % g_num[1]}_{'%1.2f' % g_num[2]}"

    def load_joint_torque_and_subs():
        tauJ_sym = cache_object('rigid_body_dynamics_linearizable', compute_tau_rbd_linearizable_parallel)
        rbd_parameters_global = list(chain.from_iterable(rigid_body_dynamics_parameters()))
        tauJ_instantiated = sp.Matrix(tauJ_sym).subs(
            sym_mat_to_subs([a, d, g, f_tcp, n_tcp, sp.Matrix(rbd_parameters_global)],
                            [a_num, d_num, g_num, f_tcp_num, n_tcp_num, np.ones_like(rbd_parameters_global)]))
        return tauJ_instantiated

    tauJ_basis = cache_object(f'./joint_torque_basis_{data_id}', load_joint_torque_and_subs)

    return tauJ_basis


def compute_observation_matrix_and_measurement_vector(data_path, regressor_base_params_instatiated, time_frame=(-np.inf, np.inf)):
    sys.setrecursionlimit(int(1e6))  # Prevents errors in sympy lambdify

    # Data
    ur5e_experiment = RobotData(data_path, delimiter=' ', desired_timeframe=time_frame, interpolate_missing_samples=True)
    qd_target = np.array([ur5e_experiment.data[f"target_qd_{j}"] for j in range(1, Njoints + 1)])
    idx_start, idx_end = find_nonstatic_start_and_end_indices(qd_target)
    q_m = np.array([ur5e_experiment.data[f"actual_q_{j}"] for j in range(1, Njoints + 1)])  # (6 x n_samples) numpy array of measured angular positions
    # plot_trajectories(t, data, joints=range(2,3))

    # Low-pass filter (smoothen) measured angular position and obtain 1st and 2nd order time-derivatives
    q_tf, qd_tf, qdd_tf = trajectory_filtering_and_central_difference(q_m, ur5e_experiment.dt_nominal, idx_start, idx_end)

    # *************************************************** PLOTS ***************************************************
    # qd_m = np.gradient(q_m, dt, edge_order=2, axis=1)
    # qdd_m = (q_m[:, 2:] - 2 * q_m[:, 1:-1] + q_m[:, :-2]) / (dt ** 2)  # two fewer indices than q and qd
    #
    # t = t[idx_start:idx_end]
    # qd_m = qd_m[:, idx_start:idx_end]
    # qdd_m = qdd_m[:, idx_start - 1:idx_end - 1]
    #
    # _, axs = plt.subplots(3, 1, sharex='all')
    # axs[0].set(ylabel='Position [rad]')
    # axs[1].set(ylabel='Velocity [rad/s]')
    # axs[2].set(ylabel='Acceleration [rad/s^2]')
    #
    # for j in range(Njoints):
    #     # Actual
    #     axs[0].plot(t, q_m[j,:], ':', color=plot_colors[j], label=f"actual_{j}")
    #     axs[1].plot(t, qd_m[j,:], ':', color=plot_colors[j], label=f"actual_{j}")
    #     axs[2].plot(t, qdd_m[j,:], ':', color=plot_colors[j], label=f"actual_{j}")
    #     # Filtered
    #     axs[0].plot(t, q_tf[j,:], '--', color=plot_colors[j], label=f"filtered_{j}")
    #     axs[1].plot(t, qd_tf[j,:], '--', color=plot_colors[j], label=f"filtered_{j}")
    #     axs[2].plot(t, qdd_f[j,:], '--', color=plot_colors[j], label=f"filtered_{j}")
    #     # Target
    #     axs[0].plot(t, data[f"target_q_{j+1}"][idx_start:idx_end], color=plot_colors[j], label=f"target_{j}")
    #     axs[1].plot(t, data[f"target_qd_{j+1}"][idx_start:idx_end], color=plot_colors[j], label=f"target_{j}")
    #     axs[2].plot(t, data[f"target_qdd_{j+1}"][idx_start:idx_end], color=plot_colors[j], label=f"target_{j}")
    #
    # for ax in axs:
    #     ax.legend()
    #
    # if not NONINTERACTIVE:
    #     plt.show()
    # *************************************************************************************************************

    i = np.array([ur5e_experiment.data[f"actual_current_{j}"] for j in range(1, Njoints + 1)]).T
    i_pf = parallel_filter(i, ur5e_experiment.dt_nominal)[idx_start:idx_end, :]
    i_pf_ds = downsample(i_pf, ur5e_experiment.dt_nominal)
    measurement_vector = i_pf_ds.flatten(order='F')  # y = [y1, ..., yi, ..., yN],  yi = [yi_{1}, ..., yi_{n_samples}]

    n_samples_ds = i_pf_ds.shape[0]  # No. of samples in downsampled data
    n_par = regressor_base_params_instatiated.shape[1]
    observation_matrix = np.zeros((Njoints * n_samples_ds, n_par))  # Initialization
    args_sym = q[1:] + qd[1:] + qdd[1:]  # List concatenation
    assert len(args_sym) == 3 * Njoints

    if load_model.lower() != 'none':
        args_sym += tauJ[1:]
        assert len(args_sym) == 4 * Njoints

    if load_model.lower() != 'none':
        tauJ_tf = compute_joint_torque_basis(q_tf, qd_tf, qdd_tf, robot_param_function=get_ur5e_parameters)
        args_num = np.concatenate((q_tf, qd_tf, qdd_tf, tauJ_tf[1:, :]))
    else:
        args_num = np.concatenate((q_tf, qd_tf, qdd_tf))

    for j in range(Njoints):  # TODO: could be computed using multiple processes
        nonzeros_j = [not elem.is_zero for elem in regressor_base_params_instatiated[j, :]]  # TODO: could be moved out of joint loop by re-writing

        # Obtain j'th row of the regressor matrix as a function of only the trajectory variables q, qd, and qdd
        regressor_base_params_instatiated_j = sp.lambdify(args_sym, regressor_base_params_instatiated[j, nonzeros_j], 'numpy')

        rows_j = regressor_base_params_instatiated_j(*args_num).transpose().squeeze()  # (1 x count(nonzeros))

        # Parallel filter and decimate/downsample rows of the observation matrix related to joint j.
        rows_j_pf_ds = downsample(parallel_filter(rows_j, ur5e_experiment.dt_nominal), ur5e_experiment.dt_nominal)

        observation_matrix[j * n_samples_ds:(j + 1) * n_samples_ds, nonzeros_j] = rows_j_pf_ds

    return observation_matrix, measurement_vector


def compute_regressor_with_instantiated_parameters(ur_param_function=get_ur5e_parameters):
    # TODO: Remove default argument being "get_ur5e_parameters". It does not make sense to have a default argument here.
    (_, d_num, a_num, _) = ur_param_function(npzeros_array)

    def to_fname(l):
        return "_".join(map(lambda s: "%1.2f" % s, l))

    data_id = f"{to_fname(d_num)}_{to_fname(a_num)}_{'%1.2f' % g_num[0]}_{'%1.2f' % g_num[1]}_{'%1.2f' % g_num[2]}"

    def load_regressor_and_subs():
        regressor_reduced = cache_object('./regressor_reduced', compute_regressor_linear_exist)
        return regressor_reduced.subs(sym_mat_to_subs([a, d, g, f_tcp, n_tcp], [a_num, d_num, g_num, f_tcp_num, n_tcp_num]))

    regressor_reduced_params = cache_object(f'./regressor_reduced_{data_id}', lambda: load_regressor_and_subs())

    return regressor_reduced_params


def compute_indices_base_exist(regressor_with_instantiated_parameters):
    """This function computes the indices for the base parameters. The number of base parameters is obtained as the
    maximum obtainable rank of the observation matrix using a set of randomly generated dummy observations for the robot
    trajectory. The specific indices for the base parameters are obtained by conducting a QR decomposition of the
    observation matrix and analyzing the diagonal elements of the upper triangular (R) matrix."""
    # TODO: Change code such that 'dummy_pos, 'dummy_vel', 'dummy_acc', and dumm_tauJ can be called with a 2D list of
    #  indices to produce a 2D list of 'dummy_pos', 'dummy_vel', and 'dummy_acc'. This way, the regressor can be called
    #  with a 2D np.array(), which will drastically speed up the computation of the dummy observation matrix.

    # The set of dummy observations
    dummy_pos = np.array([-np.pi, -0.5 * np.pi, -0.25 * np.pi, 0.0, 0.25 * np.pi, 0.5 * np.pi, np.pi])
    dummy_vel = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
    dummy_acc = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
    dummy_tauJ = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
    assert len(dummy_pos) == len(dummy_vel) == len(dummy_acc) == len(dummy_tauJ)

    max_number_of_rank_evaluations = 100
    n_rank_evals = 0  # Loop counter for observation matrix concatenations
    n_regressor_evals_per_rank_calculation = 2  # No. of regressor evaluations per rank calculation
    rank_W = []

    if load_model.lower() == 'none':
        random_idx_init = [[np.random.randint(Njoints + 1) for _ in range(Njoints)] for _ in range(3)]
        dummy_args_init = np.concatenate((dummy_pos[random_idx_init[0][:]], dummy_vel[random_idx_init[1][:]],
                                          dummy_acc[random_idx_init[2][:]]))
    else:
        random_idx_init = [[np.random.randint(Njoints + 1) for _ in range(Njoints)] for _ in range(4)]
        dummy_args_init = np.concatenate((dummy_pos[random_idx_init[0][:]], dummy_vel[random_idx_init[1][:]],
                                          dummy_acc[random_idx_init[2][:]], dummy_tauJ[random_idx_init[3][:]]))

    W_N = regressor_with_instantiated_parameters(*dummy_args_init)
    W = regressor_with_instantiated_parameters(*dummy_args_init)

    # Continue the computations while the rank of the observation matrix 'W' keeps improving
    while n_rank_evals < 3 or rank_W[-1] > rank_W[-2]:  # While the rank of the observation matrix keeps increasing
        # Generate random indices for the dummy observations
        if load_model.lower() == 'none':
            random_idx = [[[np.random.randint(Njoints + 1) for _ in range(Njoints)]
                           for _ in range(n_regressor_evals_per_rank_calculation)] for _ in range(3)]
        else:
            random_idx = [[[np.random.randint(Njoints + 1) for _ in range(Njoints)]
                           for _ in range(n_regressor_evals_per_rank_calculation)] for _ in range(4)]

        # Evaluate the regressor in a number of dummy observations and vertically stack the regressor matrices
        for i in range(n_regressor_evals_per_rank_calculation):
            if load_model.lower() == 'none':
                dummy_args = np.concatenate(
                    (dummy_pos[random_idx[0][i][:]], dummy_vel[random_idx[1][i][:]], dummy_acc[random_idx[2][i][:]]))
            else:
                dummy_args = np.concatenate((dummy_pos[random_idx[0][i][:]], dummy_vel[random_idx[1][i][:]],
                                             dummy_acc[random_idx[2][i][:]], dummy_tauJ[random_idx[3][i][:]]))

            reg_i = regressor_with_instantiated_parameters(*dummy_args)  # Each index contains a (Njoints x n_par) regressor matrix
            W_N = np.append(W_N, reg_i, axis=0)
        W = np.append(W, W_N, axis=0)

        # Evaluate rank of observation matrix
        rank_W.append(np.linalg.matrix_rank(W))
        print(f"Rank of observation matrix: {rank_W[-1]}")

        n_rank_evals += 1
        if n_rank_evals > max_number_of_rank_evaluations:
            raise Exception(f"Numerical estimation of the number of base inertial parameters did not converge within {n_rank_evals * n_regressor_evals_per_rank_calculation} regressor evaluations.")

    r = np.linalg.qr(W, mode='r')
    qr_numerical_threshold = 1e-12
    idx_is_base = abs(np.diag(r)) > qr_numerical_threshold
    idx_base = np.where(idx_is_base)[0].tolist()
    assert len(idx_base) == rank_W[-1]

    return idx_base


def evaluate_observation_matrix_cost(observation_matrix, metric="cond"):
    """This function evaluates the performance of the observation matrix as a cost, which must be low as possible."""

    if metric == "cond":
        return np.linalg.cond(observation_matrix)
    else:
        raise Exception(f"The specified metric '{metric}' is not supported.")


class Foo:
    def __init__(self):
        self.__a = 1

    def get_a(self):
        def get_a_fcn():
            aa = self.__a
            return aa
        return get_a_fcn()

    def set_a(self):
        def set_a_fcn():
            self.__a = 3
        set_a_fcn()


class LinearizationTests(TimedTest):
    def test_a(self):
        my_foo = Foo()
        print(f"my_foo.a = {my_foo.get_a()}")
        my_foo.set_a()
        print(f"my_foo.a = {my_foo.get_a()}")

    def test_joint_dynamics(self):
        compute_joint_torque_basis(np.array(1), np.array(1), np.array(1), robot_param_function=get_ur5e_parameters)
        print(1)

    def test_base_parameters(self):
        compute_parameters_base()

        sys.setrecursionlimit(int(1e6))  # Prevents errors in sympy lambdify
        args_sym = q[1:] + qd[1:] + qdd[1:]  # list concatenation
        regressor_reduced_func = sp.lambdify(args_sym, compute_regressor_with_instantiated_parameters(
            ur_param_function=get_ur5e_parameters), 'numpy')

        filename_idx = "indices_base_exist"
        filename_par = "parameters_base_exist"
        idx_base = cache_object(filename_idx, lambda: compute_indices_base_exist(regressor_reduced_func))
        par_base = cache_object(filename_par, compute_parameters_base)
        print(f"idx_base: {idx_base}")
        print(f"par_base: {par_base}")

    def test_calibration_new(self):
        # my_joint_dynamics = JointDynamics(6)
        # print(my_joint_dynamics.regressor())

        # mdh = None
        mdh_filepath = "C:/sourcecontrol/github/aurt/resources/robot_parameters/ur3e_params.csv"
        mdh = convert_file_to_mdh(mdh_filepath)
        my_robot_dynamics = RobotDynamics(mdh)
        my_robot_dynamics.regressor()

        robot_data_path = os.path.join(project_root(), 'resources', 'Dataset', 'ur5e_all_joints_same_time', 'random_motion.csv')
        t_est_val_separation = 63.0
        filename_parameters = 'parameters'
        filename_predicted_output = 'predicted_output'
        my_robot_calibration_data = RobotData(robot_data_path, delimiter=' ', desired_timeframe=(-np.inf, t_est_val_separation), interpolate_missing_samples=True)

        my_robot_calibration = RobotCalibration(my_robot_dynamics, my_robot_calibration_data)
        my_robot_calibration.calibrate(filename_parameters)
        my_robot_validation_data = RobotData(robot_data_path, delimiter=' ',
                                             desired_timeframe=(t_est_val_separation, np.inf),
                                             interpolate_missing_samples=True)
        my_robot_calibration.predict(my_robot_validation_data, filename_parameters, filename_predicted_output)

    def test_calibrate_parameters(self):
        t_est_val_separation = 63.0  # timely separation of estimation and validation datasets
        # TODO: make it possible to specify the relative portion of the dataset you want, e.g. 0.5 for half of the
        #  dataset. Also, the actual number of e.g. 52.9 does not make any sense here - I don't think the "bias"
        #  correction of the timestamps (see the function 'load_data()' in 'data_processing.py'):
        #     time_range -= time_range[0]
        #  is corrected...
        #  EDIT: Maybe the elimination of zero-velocity data plays a role(?)

        data_id = f"random_motion_{t_est_val_separation}"
        observation_matrix_file_estimation = f'./observation_matrix_estimation_{data_id}.npy'
        measurement_vector_file_estimation = f'./measurement_vector_estimation_{data_id}.npy'
        observation_matrix_file_validation = f'./observation_matrix_validation_{data_id}.npy'
        measurement_vector_file_validation = f'./measurement_vector_validation_{data_id}.npy'

        # sys.setrecursionlimit(int(1e6))  # Prevents errors in sympy lambdify
        # args_sym = q[1:] + qd[1:] + qdd[1:]  # list concatenation
        # if load_model.lower() != 'none':
            # args_sym += tauJ[1:]

        # regressor_reduced_func = sp.lambdify(args_sym, compute_regressor_with_instantiated_parameters(
        #     ur_param_function=get_ur5e_parameters), 'numpy')
        # filename = "parameter_indices_base"
        # idx_base_global = cache_object(filename, lambda: compute_indices_base_exist(regressor_reduced_func))
        # filename = 'regressor_base_with_instantiated_parameters'
        # regressor_base_params = cache_object(filename, lambda: compute_regressor_with_instantiated_parameters(
        #     ur_param_function=get_ur5e_parameters)[1:, idx_base_global])

        regressor_base_params_2 = RobotDynamics(None).regressor()

        # TODO: manually using 'store_numpy_expr' and 'load_numpy_expr' - why not use 'cache_numpy' instead?
        if not os.path.isfile(observation_matrix_file_estimation):
            # The base parameter system is obtained by passing only the 'idx_base' columns of the regressor
            root_dir = project_root()
            W_est, y_est = compute_observation_matrix_and_measurement_vector(
                os.path.join(root_dir, 'resources', 'Dataset', 'ur5e_all_joints_same_time', 'random_motion.csv'),  #'aurt/resources/Dataset/ur5e_all_joints_same_time/random_motion.csv',
                regressor_base_params_2,
                time_frame=(-np.inf, t_est_val_separation))
            W_val, y_val = compute_observation_matrix_and_measurement_vector(
                os.path.join(root_dir, 'resources', 'Dataset', 'ur5e_all_joints_same_time', 'random_motion.csv'),
                regressor_base_params_2,
                time_frame=(t_est_val_separation, np.inf))
            store_numpy_expr(W_est, observation_matrix_file_estimation)
            store_numpy_expr(y_est, measurement_vector_file_estimation)
            store_numpy_expr(W_val, observation_matrix_file_validation)
            store_numpy_expr(y_val, measurement_vector_file_validation)
        else:
            W_est = load_numpy_expr(observation_matrix_file_estimation)
            y_est = load_numpy_expr(measurement_vector_file_estimation)
            W_val = load_numpy_expr(observation_matrix_file_validation)
            y_val = load_numpy_expr(measurement_vector_file_validation)

        # ********************************************** NOT USED ATM **************************************************
        # cond = evaluate_observation_matrix_cost(W, metric="cond")
        # print(f"The condition number of the observation matrix is {cond}")
        # **************************************************************************************************************

        # sklearn fit
        OLS = LinearRegression(fit_intercept=False)
        OLS.fit(W_est, y_est)

        # Check output, i.e. evaluate i_measured - observation_matrix * p_num
        y_ols_est = OLS.predict(W_est)
        n_samples_est = round(len(y_est) / Njoints)
        n_samples_val = round(len(y_val) / Njoints)

        # Reshape measurement vector from (Njoints*n_samples x 1) to (n_samples x Njoints)
        assert n_samples_est*Njoints == len(y_ols_est) and n_samples_est*Njoints == np.shape(W_est)[0]
        y_est_reshape = np.reshape(y_est, (Njoints, n_samples_est))
        y_val_reshape = np.reshape(y_val, (Njoints, n_samples_val))
        y_est_ols_reshape = np.reshape(y_ols_est, (Njoints, n_samples_est))

        # Compute weights (the reciprocal of the estimated standard deviation of the error)
        idx_base, n_par_base, p_base = compute_parameters_base()
        residuals = y_est_reshape - y_est_ols_reshape
        residual_sum_of_squares = np.sum(np.square(residuals), axis=1)
        variance_residual = residual_sum_of_squares / (n_samples_est - np.array(n_par_base[1:]))
        # standard_deviation_residual = np.sqrt(variance_residual)

        wls_sample_weights = np.repeat(1/variance_residual, n_samples_est)

        # Weighted Least Squares solution
        WLS = LinearRegression(fit_intercept=False)
        WLS.fit(W_est, y_est, sample_weight=wls_sample_weights)
        y_wls_est = WLS.predict(W_est)
        y_wls_val = WLS.predict(W_val)
        y_wls_est_reshape = np.reshape(y_wls_est, (Njoints, n_samples_est))
        y_wls_val_reshape = np.reshape(y_wls_val, (Njoints, n_samples_val))

        # ************************************************** PLOTTING **************************************************
        mse = get_mse(y_val_reshape, y_wls_val_reshape)
        print(f"MSE: {mse}")

        t_est = np.linspace(0, n_samples_est-1, n_samples_est)*0.002  # TODO: remove hardcoded dt=0.002
        t_val = np.linspace(n_samples_est, n_samples_est + n_samples_val - 1, n_samples_val) * 0.002  # TODO: remove hardcoded dt=0.002
        fig = plt.figure()
        gs = fig.add_gridspec(2, 2, hspace=0.03, wspace=0, width_ratios=[np.max(t_est)-np.min(t_est), np.max(t_val)-np.min(t_val)])
        axs = gs.subplots(sharex='col', sharey='all')

        fig.supxlabel('Time [s]')
        fig.supylabel('Current [A]')

        # Estimation data - current
        for j in range(Njoints):
            axs[0, 0].plot(t_est, y_est_reshape[j, :].T, '-', color=plot_colors[j], linewidth=1.3, label=f'joint {j}, meas.')
            axs[0, 0].plot(t_est, y_wls_est_reshape[j, :].T, color='k', linewidth=0.6, label=f'joint {j}, pred.')
        axs[0, 0].set_xlim([t_est[0], t_est[-1]])
        axs[0, 0].set_title('Estimation')

        # Validation data - current
        for j in range(Njoints):
            axs[0, 1].plot(t_val, y_val_reshape[j, :].T, '-', color=plot_colors[j], linewidth=1.3, label=f'joint {j}, meas.')
            axs[0, 1].plot(t_val, y_wls_val_reshape[j, :].T, color='k', linewidth=0.6, label=f'joint {j}, pred. (mse: {mse[j]:.3f})')
        axs[0, 1].set_xlim([t_val[0], t_val[-1]])
        axs[0, 1].set_title('Validation')

        # Estimation data - error
        error_est = (y_est_reshape - y_wls_est_reshape)
        for j in range(Njoints):
            axs[1, 0].plot(t_est, error_est[j].T, '-', color=plot_colors[j], linewidth=1.3, label=f'joint {j+1}')
        axs[1, 0].set_xlim([t_est[0], t_est[-1]])

        # Validation data - error
        error_val = (y_val_reshape - y_wls_val_reshape)
        for j in range(Njoints):
            axs[1, 1].plot(t_val, error_val[j].T, '-', color=plot_colors[j], linewidth=1.3, label=f'joint {j+1}')
        axs[1, 1].set_xlim([t_val[0], t_val[-1]])

        # equate xtick spacing of right plots to those of left plots
        xticks_diff = axs[1, 0].get_xticks()[1] - axs[1, 0].get_xticks()[0]
        axs[1, 0].xaxis.set_major_locator(MultipleLocator(xticks_diff))
        axs[1, 1].xaxis.set_major_locator(MultipleLocator(xticks_diff))

        for ax in axs.flat:
            ax.label_outer()
        plt.setp(axs[0, 0], ylabel='Signal')
        plt.setp(axs[1, 0], ylabel='Error')

        # Legend position
        l_val = (np.max(t_val) - np.min(t_val))
        l_tot = (np.max(t_val) - np.min(t_est))
        l_val_rel = l_val / l_tot
        legend_x_position = 1 - 0.5/l_val_rel  # global center of legend as seen relative to the validation dataset
        if not NONINTERACTIVE:
            axs[0, 1].legend(loc='lower center', bbox_to_anchor=(legend_x_position, -0.022), ncol=Njoints)
            plt.show()
        # **************************************************************************************************************

    def test_linear_torques(self):
        tau_sym_linearizable = cache_object('./tau_sym_linearizable', compute_tau_sym_linearizable)

        # Checking that tau_sym_linearizable is linearizable wrt. the parameter vector
        diff_p = sp.zeros(Njoints + 1, sum(n_par_linear))
        for j in range(1, Njoints + 1):  # joint loop
            for i in range(n_par_linear[j]):  # range(n_par_j_rbd_revolute + n_par_j_joint_dynamics):
                column_idx = (j - 1) * (n_par_j_rbd_revolute + n_par_j_joint_dynamics) + i
                diff_p[j, column_idx] = sp.diff(tau_sym_linearizable[j], p_linear[j][i])
                if not NONINTERACTIVE:
                    print(f"Checking that the dynamics for joint {j} is linear with respect to the parameter {p_linear[j][i]}...")
                assert p_linear[j][i] not in diff_p[j, column_idx].free_symbols

    def test_check_reduced_regressor_linear(self):
        tau_sym_linearizable = cache_object('./tau_sym_linearizable', compute_tau_sym_linearizable)
        regressor = sp.sympify(cache_object('./regressor', compute_regressor_parallel))
        regressor_reduced, _ = cache_object('./regressor_reduced', compute_regressor_linear_exist)

        parameter_vector = sp.Matrix(sp.Matrix(p_linear)[:])

        idx_exist, _, _ = compute_parameters_linear_exist(regressor)

        check_symbolic_linear_system(tau_sym_linearizable, regressor, parameter_vector)
        check_symbolic_linear_system(tau_sym_linearizable, regressor_reduced, parameter_vector[idx_exist])


if __name__ == '__main__':
    unittest.main()
