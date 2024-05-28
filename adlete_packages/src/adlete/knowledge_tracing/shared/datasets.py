import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import pandas as pd
from adlete.training.utils import TrainOrTestStr
from EduData import get_data
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from ...utils.io import read_bytes, write_bytes
from ...utils.paths import join, make_path_absolute
from .kt_dataset import KTDataset, deserialize_dataset, serialize_dataset

type Converter = Mapping[int | str, Callable[[str], Any]] | Mapping[
    int, Callable[[str], Any]
] | Mapping[str, Callable[[str], Any]]


@dataclass
class KTDatasetMeta:
    """
    Metadata for properly preprocessing the downloaded datasets into a common format.
    """

    """
    Path of the downloaded and extracted CSV file relative to the download folder.
    """
    csv_path: str

    """
    Mapping dictionary for mapping the dataset's column names to the unifed column
    names: user_id, question_id, sequence_item_id and score.
    """
    column_mapping: Dict[str, str]

    converters: Optional[Converter] = None


DATASETS_META: Dict[str, KTDatasetMeta] = {
    "assistment-2015": KTDatasetMeta(
        csv_path="2015_100_skill_builders_main_problems/2015_100_skill_builders_main_problems.csv",
        column_mapping={
            "user_id": "user_id",
            "sequence_id": "question_id",
            "log_id": "sequence_item_id",
            "correct": "score",
        },
    ),
}


def download_dataset(dataset_id: str, download_dir: str, datasets_dir: str) -> str:
    """Downloads the dataset with the given id.

    Args:
        dataset_id (str): Id of the dataset
        download_dir (str): Download directory
        datasets_dir (str): Dataset directory, where the extracted files will be saved to.

    Returns:
        str: Path to the downloaded and extracted CSV file.
    """

    # skipping in case we already have the dataset
    new_csv_path = get_dataset_path(dataset_id, datasets_dir, "csv")
    if os.path.exists(new_csv_path):
        return new_csv_path

    # performing the actual download
    get_data(dataset_id, download_dir)

    # we're simply moving the file, instead of copying, to save space
    orig_csv_path = make_path_absolute(download_dir, DATASETS_META[dataset_id].csv_path)
    os.replace(orig_csv_path, new_csv_path)

    return new_csv_path


def get_dataset_path(dataset_id: str, datasets_dir: str, ext: str) -> str:
    if dataset_id not in DATASETS_META:
        raise Exception(f"Unsupported dataset: {dataset_id}!")

    return join(datasets_dir, f"{dataset_id}.{ext}")


def load_dataset_csv_as_df(dataset_id: str, datasets_dir: str) -> DataFrame:
    csv_path = get_dataset_path(dataset_id, datasets_dir, "csv")

    dataset_meta = DATASETS_META[dataset_id]
    column_mapping = dataset_meta.column_mapping
    use_columns = list(dataset_meta.column_mapping.keys())

    df = pd.read_csv(csv_path, usecols=use_columns, converters=dataset_meta.converters)
    return df.rename(columns=column_mapping)


def load_dataset_from_bin(bin_path: str) -> KTDataset:
    data = read_bytes(bin_path)
    dataset = deserialize_dataset(data)
    return dataset


# inspired by https://github.com/hcnoh/knowledge-tracing-collection-pytorch
def create_dataset_from_df(dataset_id: str, df: DataFrame) -> KTDataset:
    """
    Creates a knowledge tracing dataset from a pandas dataframe.

    Args:
        dataset_id (str): Id of the dataset
        df (DataFrame): Dataframe containing the dataset

    Returns:
        KTDataset: Preprocessed knowledge tracing dataset
    """
    user_ids = np.unique(df["user_id"].to_numpy())
    question_ids = np.unique(df["question_id"].to_numpy())

    user_id_index_map = {user_id: i for i, user_id in enumerate(user_ids)}
    question_id_index_map = {question_id: i for i, question_id in enumerate(question_ids)}

    question_sequences = []
    score_sequences = []

    for u in user_ids:
        user_data = df[df["user_id"] == u].sort_values("sequence_item_id")

        # Ignoring sequences with less only 1 item, since they can't be used for sequence
        # prediction.
        if user_data.shape[0] < 2:
            continue

        user_questions = np.array([question_id_index_map[q] for q in user_data["question_id"]])
        user_scores = user_data["score"].to_numpy()

        question_sequences.append(user_questions)
        score_sequences.append(user_scores)

    return KTDataset(
        dataset_id,
        user_id_index_map,
        len(question_id_index_map),
        question_id_index_map,
        question_sequences,
        score_sequences,
    )


def load_dataset(dataset_id: str, datasets_dir: str, subset: TrainOrTestStr) -> KTDataset:
    """
    Loads the training or test subset of the dataset from the file system.
    If the dataset was not preprocessed before, it will be preprocessed and the results
    cached on the file system in separate training and test set files.

    Args:
        dataset_id (str): Id of the dataset
        datasets_dir (str): Dataset directory, where the raw dataset is stored and where
                            the preprocessed subsets will be stored.
        subset (TrainOrTestStr): Subset to load, "train" or "test"

    Returns:
        KTDataset: Preprocessed knowledge tracing dataset (train or test subset)
    """
    if subset != "train" and subset != "test":
        raise Exception(f"subset must be either 'train' or 'test'. Got {subset}.")

    # we're first trying to read the cached version of the preprocessed dataset from disk
    subset_bin_path = get_dataset_path(dataset_id, datasets_dir, f"{subset}.bin")
    if os.path.exists(subset_bin_path):
        return load_dataset_from_bin(subset_bin_path)

    # alternatively we're loading the original csv an create / preprocess the dataset
    df = load_dataset_csv_as_df(dataset_id, datasets_dir)
    dataset = create_dataset_from_df(dataset_id, df)

    # For performance reasons, we're caching the preprocessed dataset on the disk, both
    # for training and testing.
    # TODO: test_size
    questions_train, questions_test, scores_train, scores_test = train_test_split(
        dataset.questions, dataset.scores
    )

    dataset_train = KTDataset(
        dataset.id,
        dataset.user_id_index_map,
        dataset.num_questions,
        dataset.question_id_index_map,
        questions_train,
        scores_train,
    )

    dataset_test = KTDataset(
        dataset.id,
        dataset.user_id_index_map,
        dataset.num_questions,
        dataset.question_id_index_map,
        questions_test,
        scores_test,
    )

    serialized_dataset_train = serialize_dataset(dataset_train)
    serialized_dataset_test = serialize_dataset(dataset_test)

    train_bin_path = get_dataset_path(dataset_id, datasets_dir, "train.bin")
    test_bin_path = get_dataset_path(dataset_id, datasets_dir, "test.bin")

    write_bytes(train_bin_path, serialized_dataset_train)
    write_bytes(test_bin_path, serialized_dataset_test)

    return dataset
