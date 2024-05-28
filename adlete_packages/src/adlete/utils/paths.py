import os
import os.path as path
import re
from typing import List


def change_extension_to(filepath: str, new_ext: str):
    root, _ext = path.splitext(filepath)
    return root + new_ext


def to_unix_path(a_path: str) -> str:
    separator = path.normpath("/")
    if separator != "/":
        a_path = re.sub(re.escape(separator), "/", a_path)
    return a_path


def make_path_absolute(root: str, rel_path: str) -> str:
    if path.isabs(rel_path):
        return to_unix_path(rel_path)

    return to_unix_path(path.normpath(path.join(root, rel_path)))


def get_relative_path(target: str, root: str) -> str:
    return to_unix_path(path.relpath(target, root))


def join(*args):
    return to_unix_path(path.join(*args))


def create_mirror_path(filepath, mirrorRoot, output_dir, ext):
    if not path.isabs(filepath):
        raise Exception("filepath must be absolute")

    if not mirrorRoot.endswith("/"):
        mirrorRoot = mirrorRoot + "/"

    rel_path = filepath.replace(mirrorRoot, "")
    return make_path_absolute(output_dir, change_extension_to(rel_path, ext))


def get_filepath_with_new_ext(file_path: str, ext: str, new_dir: str):
    base = path.basename(file_path)
    filename = change_extension_to(base, ext)
    return to_unix_path(path.join(new_dir, filename))


def child_dirs_from_dir(parent_dir: str, max_dirs: int = -1) -> List[str]:
    child_dirs = [
        child_dir
        for child_dir in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, child_dir))
    ]

    if max_dirs > -1:
        if len(child_dirs) > max_dirs:
            child_dirs = child_dirs[:max_dirs]

    return child_dirs


def get_abs_path_relative_to_file(file_path: str, relative_path: str) -> str:
    curr_dir = os.path.dirname(os.path.realpath(file_path))
    return make_path_absolute(curr_dir, relative_path)


def get_abs_path_relative_to_dir(dir_path: str, relative_path: str) -> str:
    return make_path_absolute(dir_path, relative_path)
