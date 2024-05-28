import os
from typing import Type, TypeVar

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_raw_as

from .io import read_text
from .nested_props import get_nested_prop, set_nested_prop
from .paths import get_abs_path_relative_to_dir, to_unix_path


class YAMLConfig(BaseModel):
    _dir: str


TConfig = TypeVar("TConfig", bound=YAMLConfig)


def load_config(filepath, cast_type: Type[TConfig]) -> TConfig:
    yml = read_text(filepath)
    config = parse_yaml_raw_as(cast_type, yml)
    config._dir = to_unix_path(os.path.dirname(filepath))
    return config
    # config._orig = config.copy()
    # config._dir = to_unix_path(os.path.dirname(filepath))
    # config._path = to_unix_path(filepath)
    # return cast(TConfig, config)


# TODO: make use of Annotations in the pydantic classes
def update_relative_config_paths(config: YAMLConfig, prop_paths: list[str]):
    """
    Updates all file or directory paths given in the list of property paths to
    absolute paths relative to the config file (if they are not absolute yet).
    """
    for prop_path in prop_paths:
        prop = get_nested_prop(config, prop_path)
        if prop:
            new_prop = get_abs_path_relative_to_dir(config._dir, prop)
            set_nested_prop(config, prop_path, new_prop)
