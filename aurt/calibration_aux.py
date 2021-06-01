import numpy as np


def find_first(x):
    """
    Finds first true value in the bool array x and returns the index of that value.
    """
    idx = x.view(bool).argmax() // x.itemsize
    return idx if x[idx] else -1


def find_nonstatic_start_and_end_indices(qd_target, qd_threshold=0.0001):
    """
    qd_target: (Njoints x n_samples) numpy array
    Finds start and end indices for useful (non-zero angular velocity) subset of data.
    Procedure:
    1.  Find first non-zero qd_target (starting from beginning and searching forward)
    2.  Find first non-zero qd_target (starting from end and searching backwards)
    Below is shown the angular velocity profile and the start and end indices.
                                       __
                      ^       ,___    /  \____
    qd_target [rad/s] |      /    \__/        \
                      |     /                  \
                      |____/____________________\____> sample no. [-]
                           ^idx_start           ^idx_end
    """

    idx_start = find_first(np.any(-qd_threshold > qd_target, axis=0) | np.any(qd_target > qd_threshold, axis=0))
    idx_start = max(1, idx_start)  # idx start has minimum value of 1 for estimation of double derivatives to work
    idx_end = len(qd_target[0, :]) - 1 - find_first(np.any(-qd_threshold > qd_target[:, ::-1], axis=0) | np.any(qd_target[:, ::-1] > qd_threshold, axis=0))
    assert 0 < idx_start < idx_end

    return idx_start, idx_end


def find_const_vel_start_and_end_indices(t, qd_target, qdd_threshold=0.01, t_stabilize=0.4):
    # Finds start and end indices for useful (constant angular velocity) subset of data.
    # TODO: If not qdd_target_new(idx_mid) == 0, analyse entire qdd_target_new and find the longest duration of
    #  qdd_target_new == 0 while (in addition) qd_target != 0.
    # Procedure:
    # 1.  Find center of time series data
    # 1a. Check if qdd(signal center) == 0 (constant angular velocity)
    #     From experience, the qdd_target provided by the UR controller is erroneous, thus we use:
    #          qdd_target = diff(qd_target)
    # 2.  Starting from the center of qdd_target, find the first nonzero qdd_target in each direction.
    # 3.  For the identified starting index, add a positive integer due to possible non-steady state/oscillations, which
    #     may violate the assumption of constant velocity.
    #
    # Below is shown the angular velocity profile and the start and end indices.
    #
    #                       oscillation(s)
    #                            |
    #                            v
    #                   ^       ,-_______________
    # qd_target [rad/s] |      /  :             :\
    #                   |     /   :             : \
    #                   |____/____:_____________:__\____> sample no. [-]
    #                             ^idx_start    ^idx_end
    idx_mid = t.size // 2
    dt = t[1] - t[0]
    n_stabilize = round(t_stabilize / dt)
    qdd_target = np.diff(qd_target, prepend=0.0) / dt
    assert (abs(qdd_target[idx_mid]) < qdd_threshold)
    idx_start = n_stabilize + idx_mid - find_first((-qdd_threshold > qdd_target[0:idx_mid][::-1]) | (
            qdd_target[0:idx_mid][::-1] > qdd_threshold))
    idx_end = idx_mid - 2 + find_first(
        (-qdd_threshold > qdd_target[idx_mid:-1]) | (qdd_target[idx_mid:-1] > qdd_threshold))

    assert idx_start < idx_end

    return idx_start, idx_end


def negate_negative(lst):  # return negated value of negative elements
    return [-x for x in lst if x < 0.0] or None