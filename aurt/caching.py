import os

from aurt.file_system import load_object, store_object


class Cache():
    def __init__(self):
        pass

    def get_or_cache(self, key, fun):
        raise NotImplementedError("Implemented by subclasses")

    def set(self, key, val):
        raise NotImplementedError("Implemented by subclasses")

    def get(self, key):
        raise NotImplementedError("Implemented by subclasses")

    def is_cached(self, key):
        raise NotImplementedError("Implemented by subclasses")


class PersistentPickleCache(Cache):
    def __init__(self, base_directory):
        super(PersistentPickleCache, self).__init__()
        self._base_directory = base_directory

    def get_or_cache(self, key, fun):
        """
            Caches the return value of the callable using pickle.
            """
        file = key + '.pickle'
        if os.path.exists(os.path.join(self._base_directory, file)):
            return load_object(file)
        else:
            obj = fun()
            store_object(file, obj)
            return obj

    def set(self, key, val):
        pass

    def get(self, key):
        return load_object(self._get_file_name(key))

    def _get_file_name(self, key):
        file = key + '.pickle'
        return os.path.join(self._base_directory, file)

    def is_cached(self, key):
        return os.path.exists(self._get_file_name(key))

