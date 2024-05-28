import argparse
import os
from typing import Type

from adlete.utils.config import TConfig, load_config
from adlete.utils.paths import to_unix_path


def get_config_from_cli(default_config_path: str, cast_type: Type[TConfig]) -> TConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Optional: path to main config file")

    args = parser.parse_args()

    # use main.yaml from repo or the one provided via cli
    if args.config is None:
        main_config_path = default_config_path
    else:
        main_config_path = to_unix_path(os.path.join(os.getcwd(), args.config))

    print("Loading config: {}".format(main_config_path))
    config = load_config(main_config_path, cast_type)

    return config
