import json
import os
import os.path as path

# from distutils import dir_util, file_util
from typing import Any, List

import yaml
from munch import Munch

# ================ JSON / YAML ================


def write_json(filepath: str, text: Any, encoder=None):
    os.makedirs(path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as textFile:
        textFile.write(json.dumps(text, indent=2, cls=encoder))


def read_json(filepath: str):
    with open(filepath, "r") as jsonFile:
        return json.load(jsonFile)


def read_json_as_munch(filepath: str):
    return Munch.fromDict(read_json(filepath))


def read_yaml(filepath: str) -> Any:
    with open(filepath, "r") as yamlfile:
        return yaml.safe_load(yamlfile)


def read_yaml_as_munch(filepath) -> Any:
    return Munch.fromDict(read_yaml(filepath))


def read_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8-sig") as f:
        return f.read()


def read_text_as_list(filepath: str) -> List[str]:
    with open(filepath, "r") as file:
        return file.read().splitlines()


def write_bytes(filepath: str, data: bytes):
    os.makedirs(path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as file:
        file.write(data)


def read_bytes(filepath: str) -> bytes:
    with open(filepath, "rb") as file:
        return file.read()


# def copy(src: str, dest_dir: str):
#     os.makedirs(path.dirname(dest_dir), exist_ok=True)
#     if path.isdir(src):
#         dir_util.copy_tree(src, dest_dir)
#     else:
#         file_util.copy_file(src, dest_dir)


# def copy_all(src_and_dests: List[Tuple[str, str]]):
#     for src, dest in src_and_dests:
#         copy(src, dest)


# def copy_all_relative(rel_src_and_dests: List[Tuple[str, str]], root_src: str, root_dest: str):
#     for src, dest in rel_src_and_dests:
#         copy(paths.make_path_absolute(root_src, src), paths.make_path_absolute(root_dest, dest))
