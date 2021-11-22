import os
from shutil import rmtree
from aurt.file_system import load_object, store_object


def clear_cache_dir(dir):
    if os.path.exists(dir):
        rmtree(dir)
    os.mkdir(dir)
    return dir


class Cache:
    def __init__(self):
        pass

    def get_or_cache(self, key, fun):
        raise NotImplementedError("Implemented by subclasses")


class PersistentPickleCache(Cache):
    def __init__(self, base_directory):
        super(PersistentPickleCache, self).__init__()
        self._base_directory = base_directory

    def get_or_cache(self, key, fun):
        """
        Caches the return value of the callable using pickle.
        """
        file = os.path.join(self._base_directory, key + '.pickle')
        if os.path.exists(file):
            return load_object(file)
        else:
            obj = fun()
            store_object(file, obj)
            return obj

    def _get_file_name(self, key):
        file = key + '.pickle'
        return os.path.join(self._base_directory, file)

