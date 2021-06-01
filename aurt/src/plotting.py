import numpy as np

from aurt.src.data_processing import plot_colors
from aurt.src.globals import Njoints, get_P
from aurt.src.kinematics import get_forward_kinematics
from aurt.src.num_sym_layers import npzeros_matrix, npmatrix, npcos, npsin, npvector
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from aurt.tests import NONINTERACTIVE


def build_kinematics_joint(T_i_im1, i):
    M = T_i_im1[1]
    for j in range(2, i + 1):
        M = np.dot(M, T_i_im1[j])
    return M


def draw_robot(d, a, alpha, PC,
               qs, interval_ms, axis_size=0.05, repeat=False):
    assert len(qs[0]) == Njoints + 1, "Input is expected to be a list of robot position configurations"

    # Unit vectors along each axis.
    unit_x = npvector([axis_size, 0, 0, 1])
    unit_y = npvector([0, axis_size, 0, 1])
    unit_z = npvector([0, 0, axis_size, 1])

    P = get_P(a, d, alpha, npvector, npcos, npsin)

    # PC_aug is used to make the dot product compatible with the forward kinematics conversion
    PC_aug = [npvector([PC[i][0,0], PC[i][1,0], PC[i][2,0], 1]) for i in range(0, Njoints+1)]

    def draw_robot_pos(q):
        (_, _, T_i_im1) = get_forward_kinematics(q, alpha, P, npzeros_matrix, npmatrix, npcos, npsin)

        conversion = [(build_kinematics_joint(T_i_im1, i) if i > 0 else np.identity(4)) for i in range(0, Njoints + 1)]

        # x/y/z s_frames[i] Stores the x/y/z coordinates (global frame) of the origin of frame i
        xs_frames = np.zeros(Njoints + 1)
        ys_frames = np.zeros(Njoints + 1)
        zs_frames = np.zeros(Njoints + 1)

        # Same as above, for the center of mass
        xs_pc = np.zeros(Njoints + 1)
        ys_pc = np.zeros(Njoints + 1)
        zs_pc = np.zeros(Njoints + 1)

        # Stores the x,y,z coordinates (global frame) of the unit vector x in frame i
        ux_coords_x = np.zeros(Njoints + 1)
        ux_coords_y = np.zeros(Njoints + 1)
        ux_coords_z = np.zeros(Njoints + 1)

        # Same as above, for unit vector y and z
        uy_coords_x = np.zeros(Njoints + 1)
        uy_coords_y = np.zeros(Njoints + 1)
        uy_coords_z = np.zeros(Njoints + 1)

        # Same as above, for unit vector y and z
        uz_coords_x = np.zeros(Njoints + 1)
        uz_coords_y = np.zeros(Njoints + 1)
        uz_coords_z = np.zeros(Njoints + 1)

        for i in range(0, Njoints + 1):
            p = npvector([0, 0, 0, 1])
            pos_frame = np.dot(conversion[i], p)

            xs_frames[i] = pos_frame[0, 0]
            ys_frames[i] = pos_frame[1, 0]
            zs_frames[i] = pos_frame[2, 0]

            pos_pc = np.dot(conversion[i], PC_aug[i])
            xs_pc[i] = pos_pc[0, 0]
            ys_pc[i] = pos_pc[1, 0]
            zs_pc[i] = pos_pc[2, 0]

            pos_unit_x = np.dot(conversion[i], unit_x)
            ux_coords_x[i] = pos_unit_x[0, 0]
            ux_coords_y[i] = pos_unit_x[1, 0]
            ux_coords_z[i] = pos_unit_x[2, 0]

            pos_unit_y = np.dot(conversion[i], unit_y)
            uy_coords_x[i] = pos_unit_y[0, 0]
            uy_coords_y[i] = pos_unit_y[1, 0]
            uy_coords_z[i] = pos_unit_y[2, 0]

            pos_unit_z = np.dot(conversion[i], unit_z)
            uz_coords_x[i] = pos_unit_z[0, 0]
            uz_coords_y[i] = pos_unit_z[1, 0]
            uz_coords_z[i] = pos_unit_z[2, 0]

        return (xs_frames, ys_frames, zs_frames,
                xs_pc, ys_pc, zs_pc,
                [ux_coords_x, ux_coords_y, ux_coords_z],
                [uy_coords_x, uy_coords_y, uy_coords_z],
                [uz_coords_x, uz_coords_y, uz_coords_z])

    fig = plt.figure()
    ax = p3.Axes3D(fig)

    ax.set_xlim3d([-0.9, 0.9])
    ax.set_zlim3d([-0.9, 0.9])
    ax.set_ylim3d([-0.9, 0.9])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    lines_joint = []
    pc_joint = []

    for i in range(0, Njoints):
        lines_joint.append(ax.plot([0, 1], [0, 1], [0, 1], 'o-', label=f"joint{i}", alpha=0.7)[0])

    for i in range(0, Njoints+1):
        pc_joint.append(ax.scatter([0], [0], [0], label=f"PC[{i}]"))

    arrows_x = []
    arrows_y = []
    arrows_z = []
    for j in range(0, Njoints + 1):
        arrows_x.append(ax.plot([0, 0], [0, 0], [0, 0], 'r-', alpha=0.7)[0])
        arrows_y.append(ax.plot([0, 0], [0, 0], [0, 0], 'g-', alpha=0.7)[0])
        arrows_z.append(ax.plot([0, 0], [0, 0], [0, 0], 'b-', alpha=0.7)[0])

    def update(frame):
        (xs_frames, ys_frames, zs_frames,
         xs_pc, ys_pc, zs_pc,
         ux_coords_xyz, uy_coords_xyz, uz_coords_xyz) = draw_robot_pos(qs[frame])
        for i in range(0, Njoints):
            lines_joint[i].set_data(xs_frames[i:i+2], ys_frames[i:i+2])
            lines_joint[i].set_3d_properties(zs_frames[i:i+2])

        for i in range(1, Njoints+1):
            pc_joint[i]._offsets3d = ([xs_pc[i]], [ys_pc[i]], [zs_pc[i]])

        for j in range(0, Njoints + 1):
            arrows_x[j].set_data(np.array([xs_frames[j], ux_coords_xyz[0][j]]),
                                 np.array([ys_frames[j], ux_coords_xyz[1][j]]))
            arrows_x[j].set_3d_properties(np.array([zs_frames[j], ux_coords_xyz[2][j]]))

            arrows_y[j].set_data(np.array([xs_frames[j], uy_coords_xyz[0][j]]), np.array([ys_frames[j], uy_coords_xyz[1][j]]))
            arrows_y[j].set_3d_properties(np.array([zs_frames[j], uy_coords_xyz[2][j]]))

            arrows_z[j].set_data(np.array([xs_frames[j], uz_coords_xyz[0][j]]), np.array([ys_frames[j], uz_coords_xyz[1][j]]))
            arrows_z[j].set_3d_properties(np.array([zs_frames[j], uz_coords_xyz[2][j]]))

    update(0)

    ani = animation.FuncAnimation(fig, update, frames=len(qs), interval=interval_ms, blit=False, repeat=repeat)

    ax.legend()

    return ani


def plot_trajectory(t, data, joints=range(Njoints)):
    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].set(ylabel='Position [rad]')
    axs[1].set(ylabel='Velocity [rad/s]')
    axs[2].set(ylabel='Acceleration [rad/s^2]')
    axs[2].set(xlabel='Time [s]')

    for j in joints:
        axs[0].plot(t, data[f"target_q_{j}"], label=f"q_{j}")
        axs[1].plot(t, data[f"target_qd_{j}"], label=f"qd_{j}")
        axs[2].plot(t, data[f"target_qdd_{j}"], label=f"qdd_{j}")

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.xlim((0, t[-1]))

    if not NONINTERACTIVE:
        plt.show()

    plt.close()


def plot_trajectories(t, data, joints=range(1, Njoints)):
    fig, axs = plt.subplots(3, 1, sharex='all')
    axs[0].set(ylabel='Position [rad]')
    axs[1].set(ylabel='Velocity [rad/s]')
    axs[2].set(ylabel='Acceleration [rad/s^2]')

    for j in joints:
        axs[0].plot(t, data[f"target_q_{j}"], color=plot_colors[j], label=f"target_{j}")
        axs[1].plot(t, data[f"target_qd_{j}"], color=plot_colors[j], label=f"target_{j}")
        axs[2].plot(t, data[f"target_qdd_{j}"], color=plot_colors[j], label=f"target_{j}")
        axs[0].plot(t, data[f"actual_q_{j}"], '--', color=plot_colors[j], label=f"actual_{j}")
        dt = t[1] - t[0]
        qd = np.diff(data[f"actual_q_{j}"], prepend=0.0) / dt
        qdd = np.diff(qd, prepend=0.0) / dt
        axs[1].plot(t, qd, '--', color=plot_colors[j], label=f"actual_{j}")
        axs[2].plot(t, qdd, '--', color=plot_colors[j], label=f"actual_{j}")

    for ax in axs:
        ax.legend()

    if not NONINTERACTIVE:
        plt.show()
    plt.close()


def plot_dynamics(t, data, joints=range(Njoints), custom_torque=None, custom_current=None, prefix="target"):
    n_subplots = 5 if "interpolated" not in data else 6

    fig, axs = plt.subplots(n_subplots, 1, sharex='all')
    axs[0].set(ylabel='Position [rad]')
    axs[1].set(ylabel='Velocity [rad/s]')
    axs[2].set(ylabel='Acceleration [rad/s^2]')
    axs[3].set(ylabel='Torque [Nm]')
    axs[4].set(ylabel='Current [A]')

    for j in joints:
        axs[0].plot(t, data[f"{prefix}_q_{j}"], color=plot_colors[j], label=f"{j}")
        axs[1].plot(t, data[f"{prefix}_qd_{j}"], color=plot_colors[j], label=f"{j}")
        if f"{prefix}_qdd_{j}" in data:
            axs[2].plot(t, data[f"{prefix}_qdd_{j}"], color=plot_colors[j], label=f"{j}")
        if f"{prefix}_moment_{j}" in data:
            axs[3].plot(t, data[f"{prefix}_moment_{j}"], color=plot_colors[j], label=f"UR_{j}")
        if custom_torque is not None:
            axs[3].plot(t, custom_torque[j], color=plot_colors[j], label=f"custom_{j}", linestyle='--')
        axs[4].plot(t, data[f"{prefix}_current_{j}"], color=plot_colors[j], label=f"{j}")
        if custom_current is not None:
            axs[4].plot(t, custom_current[j], color=plot_colors[j], label=f"custom_{j}", linestyle='--')

        if "interpolated" in data:
            axs[5].set(ylabel='Interpolated')
            axs[5].step(t, [1.0 if d else 0.0 for d in data["interpolated"]])

    for i in range(5):
        axs[i].legend()

    if not NONINTERACTIVE:
        plt.show()
    plt.close()
