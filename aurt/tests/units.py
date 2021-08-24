from shutil import rmtree
from aurt.file_system import from_project_root


def init_cache_dir():
    # delete cache folder before starting tests
    new_dir = from_project_root('cache')
    if new_dir.is_dir():
        rmtree(new_dir)