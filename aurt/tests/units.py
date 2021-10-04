from shutil import rmtree


def init_cache_dir(dir):
    if dir.is_dir():
        rmtree(dir)
    return dir