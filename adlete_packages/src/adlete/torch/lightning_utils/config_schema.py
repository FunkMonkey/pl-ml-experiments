from typing import Optional

from pydantic import BaseModel, Field


class TrainerConfig(BaseModel):
    """
    Configuration for Trainer class constructor
    """

    default_root_dir: Optional[str] = Field(
        None,
        title="Default Root Directory",
        description=(
            "Default path for logs and weights when no logger or "
            "lightning.pytorch.callbacks.ModelCheckpoint callback passed. On certain "
            "clusters you might want to separate where logs and checkpoints are stored. "
            "If you don’t then use this argument for convenience. Paths can be local "
            "paths or remote paths such as s3://bucket/path or hdfs://path/. "
            "Credentials will need to be set up to use remote filepaths."
        ),
    )

    max_epochs: Optional[int] = Field(
        None,
        title="Maximum number of epochs",
        description=(
            "Stop training once this number of epochs is reached. Disabled by default (None). "
            "If both max_epochs and max_steps are not specified, defaults to max_epochs = 1000. "
            "To enable infinite training, set max_epochs = -1."
        ),
    )
    min_epochs: Optional[int] = Field(
        None,
        title="Minimum number of epochs",
        description="Force training for at least these many epochs. Disabled by default (None).",
    )


class LightningConfigRoot(BaseModel):
    """Just a helper class for pydantic, so we can create our json-schema from a single root"""

    trainer: TrainerConfig
