import os

def ensure_dir_exists(path:str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
