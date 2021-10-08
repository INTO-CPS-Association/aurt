import os
from shutil import rmtree


def init_cache_dir(dir):
    if os.path.exists(dir):
        rmtree(dir)
    os.mkdir(dir)
    return dir
