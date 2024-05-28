from typing import Iterator

from torch.nn import Parameter
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from .config_schema import BaseOptimizerConfig


def create_optimizer(
    conf_optimizer: BaseOptimizerConfig, parameters: Iterator[Parameter]
) -> Optimizer:
    if not conf_optimizer.type == "Adam":
        raise Exception(f"Unsupported optimizer '{conf_optimizer.type}'")
    return Adam(parameters, lr=conf_optimizer.lr)
