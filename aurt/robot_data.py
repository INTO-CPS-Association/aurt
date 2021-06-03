import numpy as np
import csv
from itertools import compress
from aurt.globals import Njoints  # TODO: Remove 'Njoints' global definition

from aurt.file_system import safe_open

plot_colors = ['red', 'green', 'blue', 'chocolate', 'crimson', 'fuchsia', 'indigo', 'orange']

JOINT_N = "#"


class RobotData:
    """
    This class contains sampled robot data related to an experiment.
    """
    def __init__(self, file_path, delimiter, desired_timeframe=None, interpolate_missing_samples=False, ):
        self.fields = [
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
        self.__load_data(file_path, desired_timeframe=desired_timeframe, interpolate_missing_samples=interpolate_missing_samples, delimiter=delimiter)

    def __load_data(self,
                    file_path,
                    desired_timeframe=None,
                    interpolate_missing_samples=False,
                    delimiter=' '):
        """
        Loads the robot data.
        """

        self.__load_raw_csv_data(file_path=file_path, delimiter=delimiter)
        self.time -= self.time[0]  # Normalizing time range
        self.__ensure_data_consistent()

        # Trim data to the desired timeframe
        if desired_timeframe is not None:
            self.__trim_data(desired_timeframe)
            self.__ensure_data_consistent()

        self.__process_missing_samples(interpolate_missing_samples)
        self.__ensure_data_consistent()

    def __process_missing_samples(self, interpolate_missing_samples):
        """
        Corrects lost samples by interpolating data at lost samples
        """

        # If the dataset has missing samples, find;
        #   1. the time instants / sample indices having missing samples and
        #   2. the period of time missing / number of missing samples
        # and interpolate (linearly) the data for all missing samples.

        self.dt_actual = self.time[1:] - self.time[:-1]
        assert len(self.dt_actual) == (len(self.time) - 1)
        (values, counts) = np.unique(self.dt_actual, return_counts=True)
        self.dt_nominal = values[np.argmax(counts)]  # Most frequently occurring dt
        sample_is_missing = np.array([abs(number - self.dt_nominal) > 1E-6 for number in self.dt_actual])
        time_missing = self.dt_actual - self.dt_nominal
        samples_missing = [round(float(element / self.dt_nominal)) for element in time_missing]  # list of ints

        len_data_before_interpolation = len(self.time)

        assert len(samples_missing) == (len_data_before_interpolation-1)

        n_samples_to_add = sum(samples_missing)

        if any(sample_is_missing):
            print(f"Warning: {n_samples_to_add} samples are missing in this dataset.")

        if interpolate_missing_samples:
            assert "interpolated" not in self.data
            self.data["interpolated"] = np.zeros(len_data_before_interpolation, dtype=bool)
            self.data["interpolated_new"] = np.insert(sample_is_missing, 0, False)  # TODO: DELETE "self.data["interpolated"]" if self.data["interpolated"] == self.data["interpolated_new"]

            # Inserts values in data at indices of missing samples
            def insert_val(lst, idx_insert_before, v, nsamples):
                """
                Inserts a list of NaNs in data[f] at index 'idx_insert_after' having length 'samples_missing[k]'
                """
                return np.insert(lst, idx_insert_before, [v for _ in range(nsamples)])

            for k in reversed(range(len(sample_is_missing))):
                if sample_is_missing[k]:  # if a sample is missing at index k
                    idx_insert_before = k + 1
                    # Expand data and insert NaN's at locations for missing samples
                    for f in self.data:
                        if f != "interpolated":
                            self.data[f] = insert_val(self.data[f], idx_insert_before, float("nan"), samples_missing[k])
                    # Flag new insertions as interpolated
                    self.data["interpolated"] = insert_val(self.data["interpolated"], idx_insert_before, True, samples_missing[k])
                    # Also add timestamps as nan
                    self.time = insert_val(self.time, idx_insert_before, float("nan"), samples_missing[k])

            # The following two assertions ensure that the right number of samples were added.
            assert len(self.time) == len_data_before_interpolation + n_samples_to_add
            self.__ensure_data_consistent()

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
            nans, idx = nan_helper(self.time)

            def interpolate_nans(col):
                for index, value in zip(idx(nans), np.interp(idx(nans), idx(~nans), list(compress(col, ~nans)))):
                    col[index] = value

            for f in self.data:
                if f != "interpolated":
                    interpolate_nans(self.data[f])

            interpolate_nans(self.time)

    def __load_raw_csv_data(self, file_path, delimiter):
        with safe_open(file_path, mode='r') as csvFile:
            csv_reader = csv.DictReader(csvFile, delimiter=delimiter)

            self.time = np.empty(0)
            self.data = {}
            for f in self.fields:
                if JOINT_N in f:
                    for j in range(1, Njoints + 1):
                        f_j = f.replace(JOINT_N, str(j))
                        self.data[f_j] = np.empty(0)
                else:
                    self.data[f] = np.empty(0)

            for row in csv_reader:
                if not row: continue  # eliminates empty lines
                assert "timestamp" in row, f"Expected timestamp to be in the data. Perhaps you got the wrong delimiter? This is the row {row}"
                self.time = np.append(self.time, float(row["timestamp"]))
                for f in self.fields:
                    if JOINT_N in f:
                        for j in range(1, Njoints + 1):
                            f_j_out = f.replace(JOINT_N, str(j))
                            index_in = j - 1
                            f_j_in = f.replace(JOINT_N, str(index_in))
                            assert f_j_in in row, f"Problem indexing row with field {f_j_in}. Existing fields are: {row.keys()}."
                            self.data[f_j_out] = np.append(self.data[f_j_out], float(row[f_j_in]))
                    else:
                        self.data[f] = np.append(self.data[f], float(row[f]))

    def __trim_data(self, desired_timeframe):
        start_idx = 0
        while self.time[start_idx] < desired_timeframe[0]:
            start_idx += 1

        end_idx = len(self.time) - 1
        while self.time[end_idx] > desired_timeframe[1]:
            end_idx -= 1

        assert start_idx <= end_idx

        self.time = self.time[start_idx:end_idx + 1]

        for f in self.data.keys():  # crop all fields in data
            self.data[f] = self.data[f][start_idx:end_idx + 1]

    def __ensure_data_consistent(self):
        for f in self.data.keys():
            assert len(self.data[f]) == len(self.time), f"Field {f} in data is inconsistent. Expected {len(self.time)} samples. Got {len(self.data[f])} instead."