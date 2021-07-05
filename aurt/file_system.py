import os
import sys
import pickle
import numpy as np
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).parent.parent


def from_project_root(filepath):
    return project_root().joinpath(filepath)


def from_cache(filepath):
    new_dir = project_root().joinpath('cache')
    if not Path(new_dir).is_dir():
        os.mkdir(new_dir)
    return str(project_root().joinpath('cache', filepath))


def safe_open(filepath, mode='r'):
    """
    Checks if file exists and gives a friendly error in case it doesn't.
    """
    if not os.path.isfile(filepath):
        print(f"File {filepath} not found. Current dir is {os.getcwd()}")
        sys.exit(1)
    return open(filepath, mode)


def store_object(file, obj):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_object(file):
    with safe_open(file, "rb") as f:
        return pickle.load(f)


def cache_object(filename, callable):
    """
    Caches the return value of the callable using pickle.
    """
    file = filename + '.pickle'
    if os.path.exists(file):
        return load_object(file)
    else:
        obj = callable()
        store_object(file, obj)
        return obj


def store_numpy(file, nparr):
    assert file[-4:] == '.npy'
    assert isinstance(nparr, (np.ndarray, np.generic))
    with open(file, "wb") as f:
        np.save(f, nparr)


def load_numpy(file):
    with safe_open(file, "rb") as f:
        return np.load(f)


def cache_numpy(filename, callable):
    """
        Caches the return value of the callable using numpy.
    """
    file = filename + '.npy'
    if os.path.exists(file):
        return load_numpy(file)
    else:
        expr = callable()
        store_numpy(file, expr)
        return expr
