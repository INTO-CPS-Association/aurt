import numpy as np
import csv
from itertools import compress

from aurt.file_system import safe_open
from aurt.calibration_aux import find_nonstatic_start_and_end_indices

plot_colors = ['red', 'green', 'blue', 'chocolate', 'crimson', 'fuchsia', 'indigo', 'orange']

JOINT_N = "#"


class RobotData:
    """
    This class contains sampled robot data related to an experiment.
    """
    def __init__(self, file_path, delimiter, desired_timeframe=None, interpolate_missing_samples=False):
        self.n_joints = None  # TODO: automatically determine number of joints from csv data
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
        self._load_data(file_path,
                         desired_timeframe=desired_timeframe,
                         interpolate_missing_samples=interpolate_missing_samples,
                         delimiter=delimiter)
        qd_target = np.array([self.data[f"target_qd_{j}"] for j in range(1, self.n_joints + 1)])
        self.non_static_start_index, self.non_static_end_index = find_nonstatic_start_and_end_indices(qd_target)

    def _load_data(self,
                    file_path,
                    desired_timeframe=None,
                    interpolate_missing_samples=False,
                    delimiter=' '):
        """
        Loads the robot data.
        """

        self._load_raw_csv_data(file_path=file_path, delimiter=delimiter)
        self.time -= self.time[0]  # Normalizing time range
        self._ensure_data_consistent()

        # Trim data to the desired timeframe
        if desired_timeframe is not None:
            self._trim_data(desired_timeframe)
            self._ensure_data_consistent()

        self._process_missing_samples(interpolate_missing_samples)
        self._ensure_data_consistent()

    def _process_missing_samples(self, interpolate_missing_samples):
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
            print(f"Warning: {n_samples_to_add}/{len_data_before_interpolation+n_samples_to_add} ({round(n_samples_to_add/(len_data_before_interpolation+n_samples_to_add) * 100, 2)} %) samples are missing in this dataset.")

        if interpolate_missing_samples:
            assert "interpolated" not in self.data
            self.data["interpolated"] = np.zeros(len_data_before_interpolation, dtype=bool)

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
            self._ensure_data_consistent()

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

    def _load_raw_csv_data(self, file_path, delimiter):
        with safe_open(file_path, mode='r') as csvFile:
            csv_reader = csv.DictReader(csvFile, delimiter=delimiter)

            self.n_joints = len([i for i in csv_reader.fieldnames if "target_q_" in i])

            self.time = np.empty(0)
            self.data = {}
            for f in self.fields:
                if JOINT_N in f:
                    for j in range(1, self.n_joints + 1):
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
                        for j in range(1, self.n_joints + 1):
                            f_j_out = f.replace(JOINT_N, str(j))
                            index_in = j - 1
                            f_j_in = f.replace(JOINT_N, str(index_in))
                            assert f_j_in in row, f"Problem indexing row with field {f_j_in}. Existing fields are: {row.keys()}."
                            self.data[f_j_out] = np.append(self.data[f_j_out], float(row[f_j_in]))
                    else:
                        self.data[f] = np.append(self.data[f], float(row[f]))

    def _trim_data(self, desired_timeframe):
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
        
    def _ensure_data_consistent(self):
        for f in self.data.keys():
            assert len(self.data[f]) == len(self.time), f"Field {f} in data is inconsistent. Expected {len(self.time)} samples. Got {len(self.data[f])} instead."

    def plot_missing_samples(self):
        import matplotlib.pyplot as plt
        plt.plot(self.data["interpolated"])
        plt.show()

    def plot_target_trajectory(self):
        q_m = np.array([self.data[f"actual_q_{j}"] for j in range(1, self.n_joints + 1)])
        qd_m = np.gradient(q_m, self.dt_nominal, edge_order=2, axis=1)
        qdd_m = (q_m[:, 2:] - 2 * q_m[:, 1:-1] + q_m[:, :-2]) / (self.dt_nominal ** 2)  # two fewer indices than q and qd

        # q,qd,qdd = central_finite_difference_and_crop(order=2, idx_start, idx_end)

        t = self.data["timestamp"]
        qd_m = qd_m[:, 1:-2]

        if idx_end == -1:
            qdd_end_idx = qd_tf.shape[1]
        else:
            qdd_end_idx = idx_end-1

        qdd_tf = qdd_tf[:, idx_start - 1:qdd_end_idx]
        qdd_m = qdd_m[:, -1:-1], np.append(qdd_m, None, axis=1)

        import matplotlib.pyplot as plt

        _, axs = plt.subplots(3, 1, sharex='all')
        axs[0].set(ylabel='Position [rad]')
        axs[1].set(ylabel='Velocity [rad/s]')
        axs[2].set(ylabel='Acceleration [rad/s^2]')

        for j in range(self.n_joints):
            # Actual
            axs[0].plot(t, q_m[j, :], ':', color=plot_colors[j], label=f"actual_{j}")
            axs[1].plot(t, qd_m[j, :], ':', color=plot_colors[j], label=f"actual_{j}")
            axs[2].plot(t, qdd_m[j, :], ':', color=plot_colors[j], label=f"actual_{j}")
            # Target
            axs[0].plot(t, self.data[f"target_q_{j+1}"], color=plot_colors[j], label=f"target_{j}")
            axs[1].plot(t, self.data[f"target_qd_{j+1}"], color=plot_colors[j], label=f"target_{j}")
            axs[2].plot(t, self.data[f"target_qdd_{j+1}"], color=plot_colors[j], label=f"target_{j}")

        for ax in axs:
            ax.legend()


        for j in range(self.n_joints):
            plt.plot(self.data["timestamp"], self.data[f"target_q_{j+1}"], color=plot_colors[j])

        plt.show()
