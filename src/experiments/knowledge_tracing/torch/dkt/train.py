# -------------------------------------------------------------------------##
# You can run this python script directly!                                 ##
#                                                                          ##
# If you want to use a different configuration file, then specify it via   ##
# `--config path/to/config.yaml`.                                          ##
#                                                                          ##
# This is an implementation of deep knowledge tracing.                     ##
#                                                                          ##
# -------------------------------------------------------------------------##


from adlete.knowledge_tracing.shared.datasets import load_dataset
from adlete.knowledge_tracing.torch.dataset import get_torch_data_loaders
from adlete.knowledge_tracing.torch.dkt.dkt import DKTModel
from adlete.knowledge_tracing.torch.kt_lit_model import KTLitModel
from adlete.utils.config import update_relative_config_paths
from adlete.utils.paths import get_abs_path_relative_to_file
from lightning import Trainer

from experiments.cli_configured import get_config_from_cli
from experiments.knowledge_tracing.torch.dkt.config_schema import DKTConfigFile

# -- loading the config --
config_path = get_abs_path_relative_to_file(__file__, "config.yaml")
config = get_config_from_cli(config_path, DKTConfigFile)
update_relative_config_paths(config, ["dataset.datasets_dir", "training.trainer.default_root_dir"])

# -- loading the dataset --
raw_dataset = load_dataset(config.dataset.dataset_id, config.dataset.datasets_dir, "train")
training_loader, validation_loader = get_torch_data_loaders(raw_dataset, config.training.batch_size)

# we can get num_questions from the data
config.model.num_questions = raw_dataset.num_questions

# -- starting the training --
trainer = Trainer(**config.training.trainer.__dict__)

base_model = DKTModel(config.model)
lit_model = KTLitModel(base_model, config.training.optimization.optimizer)

trainer.fit(lit_model, training_loader, validation_loader)
