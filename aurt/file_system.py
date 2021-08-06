import os
import sys
import pickle
import numpy as np
from pathlib import Path
import pandas as pd


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
    # assert isinstance(nparr, (np.ndarray, np.generic))
    print(f"file: {file}")
    print(f"nparr: {nparr}")
    with open(file, "wb") as f:
        np.save(f, nparr)

def store_csv(file, data):
    assert file[-4:] == '.csv'
    df = pd.DataFrame(data)
    df.to_csv(file, index=False, header=False)

def load_numpy(file):
    with safe_open(file, "rb") as f:
        return np.load(f)

def load_csv(file):
    return pd.read_csv(file, header=None)

def cache_numpy(file, callable):
    """
        Caches the return value of the callable using numpy.
    """
    if not file[-4:] == '.npy':
        file = file + '.npy'

    if os.path.exists(file):
        return load_numpy(file)
    else:
        expr = callable()
        store_numpy(file, expr)
        return expr



def cache_csv(file, callable):
    if not file[-4:] == '.csv':
        file = file + '.csv'

    if os.path.exists(file):
        return load_csv(file)
    else:
        expr = callable()
        store_csv(file, expr)
        return expr