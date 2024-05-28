from adlete.knowledge_tracing.shared.datasets import download_dataset
from adlete.utils.paths import get_abs_path_relative_to_dir, get_abs_path_relative_to_file

from experiments.cli_configured import get_config_from_cli
from experiments.knowledge_tracing.datasets.config_schema import DownloadDatasetsConfigFile

# -------------------------------------------------------------------------##
# You can run this python script directly!                                 ##
# If you want to use a different configuration file, then specify it via   ##
# `--config path/to/config.yaml`.                                          ##
# -------------------------------------------------------------------------##


# loading the config
config_path = get_abs_path_relative_to_file(__file__, "config.yaml")
config = get_config_from_cli(config_path, DownloadDatasetsConfigFile)

# starting the training
download_dir = get_abs_path_relative_to_dir(config._dir, config.download_dir)
datasets_dir = get_abs_path_relative_to_dir(config._dir, config.datasets_dir)


for dataset_id in config.datasets:
    print(download_dataset(dataset_id, download_dir, datasets_dir))
