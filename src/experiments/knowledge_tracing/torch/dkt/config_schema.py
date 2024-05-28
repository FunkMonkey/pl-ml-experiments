from adlete.knowledge_tracing.torch.dkt.dkt import DKTModelConfig
from adlete.training.config_schema import TrainingConfig
from adlete.utils.config import YAMLConfig

from ...datasets.config_schema import KTDatasetConfig


class DKTConfigFile(YAMLConfig, extra="forbid"):
    dataset: KTDatasetConfig
    model: DKTModelConfig
    training: TrainingConfig
