from typing import Literal, Union

from pydantic import BaseModel, Field

from ..torch.lightning_utils.config_schema import TrainerConfig


class BaseOptimizerConfig(BaseModel):
    type: str = Field(title="Optimizer Type", description="Type of optimizer to use, e.g. 'Adam'.")
    lr: float = Field(0.001, title="Learning Rate")


class AdamOptimizerConfig(BaseOptimizerConfig):
    type: Literal["Adam"] = Field(
        "Adam", title="Optimizer Type", description="Type of optimizer to use, fixed to 'Adam'."
    )


class CustomOptimizerConfig(BaseOptimizerConfig):
    type: Literal["Custom"] = Field(
        "Custom", title="Optimizer Type", description="Type of optimizer to use, fixed to 'Custom'."
    )


class OptimizationConfig(BaseModel):
    optimizer: Union[CustomOptimizerConfig, AdamOptimizerConfig] = Field(
        title="Optimizer", description="Configuration of the optimizer.", discriminator="type"
    )


class TrainingConfig(BaseModel):
    trainer: TrainerConfig = Field(
        title="Lightning Trainer", description="Configuration of the Lightning Trainer class"
    )
    optimization: OptimizationConfig = Field(
        title="Optimization", description="Configuration of the optimization step."
    )
    batch_size: int = Field(title="Batch Size", description="The size of a single batch.")


class TrainingConfigRoot(BaseModel):
    """Just a helper class for pydantic, so we can create our json-schema from a single root"""

    training: TrainingConfig
