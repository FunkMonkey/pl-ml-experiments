from typing import List

from adlete.utils.config import YAMLConfig
from pydantic import BaseModel, Field


class KTDatasetConfig(BaseModel):
    datasets_dir: str = Field(
        title="Datasets Directory", description="Directory, in which the datasets are located."
    )
    dataset_id: str = Field(title="Dataset Id", description="Id of the dataset to use.")


class DownloadDatasetsConfigFile(YAMLConfig):
    datasets: List[str] = Field(title="Datasets", description="Datasets to download.")
    download_dir: str = Field(title="Download Directory")
    datasets_dir: str = Field(
        title="Datasets Directory",
        description="Directory, where the preprocessed datasets will be saved.",
    )


class KTDatasetConfigRoot(BaseModel):
    """Just a helper class for pydantic, so we can create our json-schema from a single root"""

    kt_dataset: KTDatasetConfig
    download_datasets: DownloadDatasetsConfigFile
