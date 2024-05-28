from dataclasses import dataclass
from typing import Dict, List

import dill
from numpy import ndarray


@dataclass
class KTDataset:
    """Preprocessed knowledge tracing dataset"""

    id: str
    user_id_index_map: Dict[int, int]
    num_questions: int
    question_id_index_map: Dict[int, int]
    questions: List[ndarray]
    scores: List[ndarray]


# TODO: check if serialization to JSON via pydantic works, especially for the numpy arrays
def serialize_dataset(dataset: KTDataset) -> bytes:
    return dill.dumps(dataset)


def deserialize_dataset(data: bytes) -> KTDataset:
    return dill.loads(data)
