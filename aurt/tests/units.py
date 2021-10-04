from shutil import rmtree
from aurt.file_system import from_project_root


def init_cache_dir(dir):
    if dir.is_dir():
        rmtree(dir)
    return dir