import os
import sys
import pickle
import numpy as np
from pathlib import Path


def project_root() -> Path:
    """TODO: Validate whether this is a valid approach."""
    return Path(__file__).parent.parent


def safe_open(filepath, mode='r'):
    """
    Checks if file exists and gives a friendly error in case it doesn't.
    """
    if not os.path.isfile(filepath):
        print(f"File {filepath} not found. Current dir is {os.getcwd()}")
        sys.exit(1)
    return open(filepath, mode)


def store_object(expr, file):
    with open(file, "wb") as f:
        pickle.dump(expr, f)


def load_object(file):
    with safe_open(file, "rb") as f:
        return pickle.load(f)


def cache_object(filename, callable):
    """
    This function implements caching of the return value of the callable.
    """
    file = filename + '.pickle'
    if os.path.exists(file):
        return load_object(file)
    else:
        obj = callable()
        store_object(obj, file)
        return obj


def cache_object_new(filename, callable, supplied_arguments=None):
    """
    This function implements caching of the return value of the callable depending on the specific arguments supplied to
    the callable. The arguments are to be supplied in a dictionary.
    """
    hash_value_supplied_arguments = hash(supplied_arguments)
    file_data = filename + f'_data_{hash_value_supplied_arguments}.pickle'
    file_args = filename + f'_args_{hash_value_supplied_arguments}.pickle'
    if os.path.exists(file_data):  # If the data file exists (the supplied arguments have the same hash value as an existing file)
        if supplied_arguments is not None:  # if argument(s) are supplied
            if os.path.exists(file_args):  # stored and supplied arguments are probably, but not necessarily, equal
                stored_arguments = load_object(file_args)
                hash_value_stored_arguments = hash(stored_arguments)
                if len(stored_arguments) == len(supplied_arguments) and stored_arguments == supplied_arguments:  # check content of the supplied arguments against the stored arguments
                    return load_object(file_data)
                else:  # supplied arguments differ from the stored arguments EVEN THOUGH they have the same hash value

                    # Generate unique filename
                    i = 0
                    file_data_new = file_data
                    while os.path.exists(file_data_new):
                        file_data_new = file_data + f'_{i}'
                        i += 1

                    obj = callable()
                    store_object(obj, file_data_new)
                    store_object(supplied_arguments, file_args)
            else:  # file with arguments doesn't exist
                store_object(supplied_arguments, file_args)
                return load_object(file_data)
        else:  # arguments not provided, i.e. callable() doesn't depend on any arguments
            return load_object(file_data)
    else:
        obj = callable()
        store_object(obj, file_data)
        store_object(supplied_arguments, file_args)
        return obj


def get_unique_filename(filename):
    i = 0
    filename_unique = filename
    while os.path.exists(filename_unique):
        filename_unique = filename + f'_{i}'
        i += 1
    return filename_unique

def store_numpy_expr(nparr, file):
    assert file[-4:] == '.npy'
    assert isinstance(nparr, (np.ndarray, np.generic))
    with open(file, "wb") as f:
        np.save(f, nparr)


def load_numpy_expr(file):
    with safe_open(file, "rb") as f:
        return np.load(f)


def cache_numpy(file, callable):
    if os.path.exists(file):
        return load_numpy_expr(file)
    else:
        expr = callable()
        store_numpy_expr(expr, file)
        return expr