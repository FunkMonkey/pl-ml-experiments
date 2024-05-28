from typing import Optional

import torch
from numpy import ndarray
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

from ..shared.kt_dataset import KTDataset

type TorchKTBatch = tuple[Tensor, Tensor, Tensor]


class TorchKTDataset(Dataset):
    """Pytorch wrapper for KTDataset"""

    def __init__(self, dataset: KTDataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset.questions[index], self.dataset.scores[index]

    def __len__(self):
        return len(self.dataset.questions)


# TODO: add configuration, e.g. including:
# - pre- vs. post padding (maybe not necessary due to the masking)
def collate_fn(data: list[tuple[ndarray, ndarray]]) -> TorchKTBatch:
    """
    Collate function for transforming the data of the raw KTDataset into
    pytorch tensors.

    Args:
        data (list[tuple[ndarray, ndarray]]): Batch of KTDataset data

    Returns:
        TorchKTBatch: Batch of pytorch tensors tuples (padded questions, padded scores, mask).
                      Shape will depend on the max sequence length in the batch.
    """
    padding_value = -1

    questions = [torch.from_numpy(elem[0]) for elem in data]
    scores = [torch.from_numpy(elem[1]) for elem in data]

    # TODO: make binarization optional
    # TODO: add alternative binarization options
    scores = [torch.where(item_scores >= 0.5, 1.0, 0.0).int() for item_scores in scores]

    # padding, so we can put all in one batch
    padded_questions = pad_sequence(questions, batch_first=True, padding_value=padding_value)
    padded_scores = pad_sequence(scores, batch_first=True, padding_value=padding_value)
    mask = padded_questions != padding_value

    # the lengths of the sequences may be useful for masking
    # seq_lengths = torch.IntTensor([len(elem[0]) for elem in data])

    return padded_questions, padded_scores, mask


def get_torch_data_loaders(
    dataset: KTDataset, batch_size: Optional[int] = None
) -> tuple[DataLoader, DataLoader]:
    """Creates the training and validation data loaders

    Args:
        dataset (KTDataset): Raw knowledge tracing dataset
        batch_size (Optional[int]): Batch_size

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation data loader
    """
    torch_dataset = TorchKTDataset(dataset)
    train_data, validation_data = random_split(torch_dataset, (0.9, 0.1))

    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        DataLoader(validation_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
    )
