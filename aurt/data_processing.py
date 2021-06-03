<<<<<<< HEAD
=======
import csv
from itertools import compress
from math import pi

import numpy as np
import pandas as pd


# Taken from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
from aurt.file_system import safe_open
from aurt.globals import Njoints

plot_colors = ['red', 'green', 'blue', 'chocolate', 'crimson', 'fuchsia', 'indigo', 'orange']

JOINT_N = "#"


ur5e_fields = [
            f"timestamp",
            f"target_q_{JOINT_N}",
            f"target_qd_{JOINT_N}",
            f"target_qdd_{JOINT_N}",
            f"target_current_{JOINT_N}",
            f"actual_current_{JOINT_N}",
            f"target_moment_{JOINT_N}",
            f"actual_q_{JOINT_N}",
            f"actual_qd_{JOINT_N}",
        ]


def convert_file_to_mdh(filename):
    if filename[-3:] == "csv":
        df = pd.read_csv(filename)
    df = df.fillna(value='None')
    d = [float(d) if d != 'None' else None for d in df.d]
    a = [float(a) if a != 'None' else None for a in df.a]
    alpha = []
    for alpha_i in df.alpha:
        alpha.append(input_with_pi_to_float(alpha_i))
    return d, a, alpha


def input_with_pi_to_float(input):
    if input == 'None':
        return None
    elif isinstance(input, str) and "pi" in input:
        input = input.split("/")
        # Case: no / in input, i.e. either pi or a number, or -pi
        if len(input) == 1: # this means no / is in input
            if "pi" == input[0]:
                return pi
            elif "-pi" == input[0]:
                return -pi
            else:
                return float(input)
        # Case: / in input, either pi/2, pi/4, -pi/2, -pi/8, or number/number
        elif len(input) == 2:
            if "-pi" == input[0]:
                return -pi/float(input[1])
            elif "pi" == input[0]:
                return pi/float(input[1])
        else:
            print(f"Whoops, len of input is greater than 2: {len(input)}")
    else:
        return float(input)


def load_raw_csv_data(file_path, fields, sample_step, delimiter):
    time_range = []
    data = {}
    for f in fields:
        if JOINT_N in f:
            # TODO: Don't use hardcoded value of 'Njoints'
            for j in range(1, Njoints + 1):
                f_j = f.replace(JOINT_N, str(j))
                data[f_j] = []
        else:
            data[f] = []

    i = 0
    with safe_open(file_path, mode='r') as csvFile:
        csvReader = csv.DictReader(csvFile, delimiter=delimiter)

        for row in csvReader:
            if not row: continue  # eliminates empty lines
            # TODO: Why not do "if row: i += 1 [...]" and omit the line "if not row: continue"?
            i += 1
            if i >= sample_step:
                # TODO: align 'timestamp' naming with user data
                assert "timestamp" in row, f"Expected timestamp to be in the data. Perhaps you got the wrong delimiter? This is the row {row}"
                time_range.append(float(row["timestamp"]))
                for f in fields:
                    if JOINT_N in f:
                        for j in range(1, Njoints + 1):
                            f_j_out = f.replace(JOINT_N, str(j))
                            index_in = j - 1
                            f_j_in = f.replace(JOINT_N, str(index_in))
                            assert f_j_in in row, f"Problem indexing row with field {f_j_in}. Existing fields are: {row.keys()}."
                            data[f_j_out].append(float(row[f_j_in]))
                    else:
                        data[f].append(float(row[f]))
                i = 0
    return time_range, data


def trim_data(time_range, data, desired_timeframe):
    start_idx = 0
    while time_range[start_idx] < desired_timeframe[0]:
        start_idx += 1

    end_idx = len(time_range) - 1
    while time_range[end_idx] > desired_timeframe[1]:
        end_idx -= 1

    assert start_idx <= end_idx

    time_range = time_range[start_idx:end_idx + 1]  # crop time_range

    for f in data.keys():  # crop all fields in data
        data[f] = data[f][start_idx:end_idx + 1]

    return time_range, data


def process_missing_samples(time_range, data, interpolate_missing_samples):
    """
    Corrects lost samples by;
      1. interpolating data at lost samples
         Q:  how to deal with position, velocity, and acceleration?
         A1: interpolate acceleration and derive/compute velocity and position. Position and velocity may not fit in the end!
         A2: interpolate position, velocity, and acceleration
         A3: Fit position, velocity, and acceleration in a least squares fashion (evaluated at X samples before and after missing samples start)
    """

    # If the dataset has missing samples, find;
    #   1. the time instants / sample indices having missing samples and
    #   2. the period of time missing / number of missing samples
    # and interpolate (linearly) the data for all missing samples.

    dt_actual = time_range[1:] - time_range[:-1]
    assert len(dt_actual) == (len(time_range) - 1)
    (values, counts) = np.unique(dt_actual, return_counts=True)
    dt_nominal = values[np.argmax(counts)]  # Most frequently occurring dt
    sample_is_missing = [abs(number - dt_nominal) > 1E-6 for number in dt_actual]
    time_missing = dt_actual - dt_nominal
    samples_missing = [round(float(element / dt_nominal)) for element in time_missing]  # list of ints

    len_data_before_interpolation = len(time_range)

    assert len(samples_missing) == (len_data_before_interpolation-1)

    n_samples_to_add = sum(samples_missing)

    if any(sample_is_missing):
        print(f"Warning: {n_samples_to_add} samples are missing in this dataset.")

    if interpolate_missing_samples:

        # This is so that we don't touch the original data passed to this function.
        # If we don't do this, this function will change the original data, but not change the original time_range.
        # Such behavior would be inconsistent.
        data = data.copy()

        assert "interpolated" not in data
        data["interpolated"] = np.zeros(len_data_before_interpolation, dtype=bool)

        # Inserts values in data at indices of missing samples
        def insert_val(list, idx_insert_before, v, nsamples):
            """
            Inserts a list of NaNs in data[f] at index 'idx_insert_after' having length 'samples_missing[k]'
            """
            return np.insert(list, idx_insert_before, [v for _ in range(nsamples)])

        for k in reversed(range(len(sample_is_missing))):
            if sample_is_missing[k]:  # if a sample is missing at index k
                idx_insert_before = k + 1
                # Expand data and insert NaN's at locations for missing samples
                for f in data:
                    if f != "interpolated":
                        data[f] = insert_val(data[f], idx_insert_before, float("nan"), samples_missing[k])
                # Flag new insertions as interpolated
                data["interpolated"] = insert_val(data["interpolated"], idx_insert_before, True, samples_missing[k])
                # Also add timestamps as nan
                time_range = insert_val(time_range, idx_insert_before, float("nan"), samples_missing[k])

        # The following two assertions ensure that the right number of samples were added.
        assert len(time_range) == len_data_before_interpolation + n_samples_to_add
        ensure_data_consistent(time_range, data)

        def nan_helper(y):
            """Helper to handle indices and logical indices of NaNs.

            Input:
                - y, 1d numpy array with possible NaNs
            Output:
                - nans, logical indices of NaNs
                - index, a function, with signature indices = index(logical_indices),
                  to convert logical indices of NaNs to 'equivalent' indices
            Example:
                # linear interpolation of NaNs
                nans, x = nan_helper(y)
                y[nans] = np.interp(x(nans), x(~nans), y[~nans])
            """

            return np.isnan(y), lambda z: z.nonzero()[0]

        # Interpolate NaN values
        nans, idx = nan_helper(time_range)

        def interpolate_nans(col):
            for index, value in zip(idx(nans), np.interp(idx(nans), idx(~nans), list(compress(col, ~nans)))):
                col[index] = value

        for f in data:
            if f != "interpolated":
                interpolate_nans(data[f])

        interpolate_nans(time_range)

    return time_range, data


def load_data(file_path,
              fields,
              desired_timeframe=None,
              sample_step=1,
              interpolate_missing_samples=False,
              delimiter=' '):
    """
    Loads the UR robot data.
    """

    time_range, data = load_raw_csv_data(file_path, fields, sample_step, delimiter)

    ensure_data_consistent(time_range, data)

    # Compute derived data
    # TODO: Delete these following augmented joint data
    data["target_q_0"] = np.zeros(len(time_range))
    data["target_qd_0"] = np.zeros(len(time_range))
    data["target_qdd_0"] = np.zeros(len(time_range))
    data["target_current_0"] = np.zeros(len(time_range))
    data["target_moment_0"] = np.zeros(len(time_range))

    # Normalizing timerange
    time_range = np.array(time_range)
    time_range -= time_range[0]

    ensure_data_consistent(time_range, data)

    # Trim data to the desired timeframe
    if desired_timeframe is not None:
        time_range, data = trim_data(time_range, data, desired_timeframe)
        ensure_data_consistent(time_range, data)

    time_range, data = process_missing_samples(time_range, data, interpolate_missing_samples)

    # Make sure all signals have the same size.
    ensure_data_consistent(time_range, data)

    return time_range, data


def ensure_data_consistent(time_range, data):
    signal_length = len(time_range)
    for f in data.keys():
        assert len(data[f]) == signal_length, f"Field {f} in data is inconsistent. Expected {signal_length} samples. Got {len(data[f])} instead."
>>>>>>> 661ff98a0978b8ff4df920f640cc6011ddfa2462
