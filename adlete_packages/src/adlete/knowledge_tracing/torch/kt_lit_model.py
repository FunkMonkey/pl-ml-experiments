from typing import Tuple

import lightning
import torch
from torch import Tensor, nn

# from torch.masked import as_masked_tensor
from torch.nn.functional import binary_cross_entropy, one_hot

from ...training.config_schema import BaseOptimizerConfig
from ...training.optimization import create_optimizer
from .dataset import TorchKTBatch


def get_truncated_and_shifted_tensors(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    return tensor[..., :-1], tensor[..., 1:]


def reduce_by_one_hot(tensor: Tensor, one_hot_vec: Tensor) -> Tensor:
    """
    Reduces the last dimension of the tensor by picking the values according
    to the one_hot_vector.
    """
    return (tensor * one_hot_vec).sum(-1)


class KTLitModel(lightning.LightningModule):
    """
    Lightning module for knowledge tracing algorithms.

    Contains boilerplate code for transforming the questions and scores and
    calculating the loss.
    """

    def __init__(self, model: nn.Module, conf_optimizer: BaseOptimizerConfig):
        """
        Constructor

        Args:
            model (nn.Module): _description_
            conf_optimizer (BaseOptimizerConfig): _description_
        """
        super().__init__()
        self.model = model
        self.conf_optimizer = conf_optimizer

    def perform_step(self, batch: TorchKTBatch, metric_prefix: str) -> Tensor:
        """
        Forward propagation and loss calculationg

        Args:
            batch (TorchKTBatch): The batch used for training / validating.
            metric_prefix (str): Prefix for the metric (e.g. train or test)

        Returns:
            Tensor: Loss
        """
        questions, scores, mask = batch
        # questions_masked = as_masked_tensor(questions, mask)
        # scores_masked = as_masked_tensor(scores, mask)
        questions_masked = questions * mask
        scores_masked = scores * mask

        questions_truncated, questions_shifted = get_truncated_and_shifted_tensors(questions_masked)
        scores_truncated, scores_shifted = get_truncated_and_shifted_tensors(scores_masked)

        # Predictions of the score for every question type
        # We use the truncated inputs, since we need the last elements as the ground truth.
        score_predictions = self.model(questions_truncated.long(), scores_truncated.long())

        # Reducing the predictions to the same shape as the scores for calculating the loss.
        # Due to the multiplication in reduce_by_one_hot multiplication, the backpropagation
        # will not update the gradients of unrelated questions. This seems a little simpler
        # compared to extending scores_shifted with the question type dimension to match
        # the model output.
        questions_shifted_one_hot = one_hot(questions_shifted.long(), 100)
        reduced_predictions = reduce_by_one_hot(score_predictions, questions_shifted_one_hot)

        mask_shifted = mask[..., 1:]
        predictions_1d = torch.masked_select(reduced_predictions, mask_shifted)
        scores_shifted_1d = torch.masked_select(scores_shifted, mask_shifted).float()

        loss = binary_cross_entropy(predictions_1d, scores_shifted_1d)

        self.log(f"{metric_prefix}_loss", loss)

        return loss

    def training_step(self, batch: TorchKTBatch) -> Tensor:
        return self.perform_step(batch, "train")

    def validation_step(self, batch: TorchKTBatch) -> Tensor:
        return self.perform_step(batch, "val")

    def configure_optimizers(self):
        return create_optimizer(self.conf_optimizer, self.parameters())
