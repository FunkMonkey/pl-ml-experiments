# This module is heavily inspired by the knowledge-tracing-collection-pytorch
# https://github.com/hcnoh/knowledge-tracing-collection-pytorch/blob/main/models/dkt.py
# See also the original Deep Knowledge Tracing paper: https://dl.acm.org/doi/10.5555/2969239.2969296


import pydantic
from torch import Tensor, nn, sigmoid, stack


class DKTModelConfig(pydantic.BaseModel):
    num_questions: int
    use_embeddings: bool
    embedding_size: int
    lstm_hidden_size: int


class DKTModel(nn.Module):
    """
    Deep Knowledge Tracing module
    """

    def __init__(self, config: DKTModelConfig):
        super().__init__()

        self.config = config

        if config.use_embeddings:
            self.embedding_layer = nn.Embedding(config.num_questions * 2, config.embedding_size)
            lstm_input_size = config.embedding_size
        else:
            lstm_input_size = 2

        self.lstm_layer = nn.LSTM(lstm_input_size, config.lstm_hidden_size, batch_first=True)
        self.output_layer = nn.Linear(config.lstm_hidden_size, config.num_questions)
        self.dropout_layer = nn.Dropout()

    def forward(self, questions: Tensor, scores: Tensor):
        """
        Forward pass

        Args:
            questions (Tensor[batch, n]): Sequence of questions
            scores (Tensor[batch, n]): Sequence of scores

        Returns:
            Tensor[batch, n, num_questions]: Prediction for score per question type
        """
        if self.config.use_embeddings:
            # Combining questions and scores into a single vector used for embeddings.
            # This will only work for binarized scores!
            # TODO: create a check for binarized scores
            x = questions + self.config.num_questions * scores
            x = self.embedding_layer(x)
        else:
            # just using both features "as is"
            x = stack((questions, scores), dim=2).float()

        lstm_hidden, _ = self.lstm_layer(x)
        y = self.output_layer(lstm_hidden)
        y = self.dropout_layer(y)
        y = sigmoid(y)

        return y
