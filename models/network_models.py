"""Network models"""

import math
from typing import Tuple, Callable

import torch
import torch.nn.functional

from layers.embedding import EmbeddingLayer
from layers.encoding import EncodingLayer
from layers.reading import ReadingLayer
from layers.writing import WritingLayer
from models.neuron_models import NeuronModel


class MemorizingAssociations(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int, num_embeddings: int, embedding_size: int, memory_size: int,
                 num_time_steps: int, readout_delay: int, tau_trace: float, plasticity_rule: Callable,
                 dynamics: NeuronModel) -> None:
        super().__init__()
        self.readout_delay = readout_delay

        self.input_layer = torch.nn.Linear(input_size, embedding_size, bias=False)
        self.embedding_layer = EmbeddingLayer(num_embeddings, embedding_size)
        self.encoding_layer = EncodingLayer(embedding_size, embedding_size, False, False, num_time_steps, dynamics)
        self.writing_layer = WritingLayer(2*embedding_size, memory_size, plasticity_rule, tau_trace, dynamics)
        self.reading_layer = ReadingLayer(embedding_size, memory_size, readout_delay, dynamics)
        self.output_layer = torch.nn.Linear(memory_size, output_size, bias=False)

    def forward(self, features: torch.Tensor, labels: torch.Tensor, query: torch.Tensor) -> Tuple:
        features_embedded = self.input_layer(features)
        labels_embedded = self.embedding_layer(labels)
        query_embedded = self.input_layer(query)

        features_encoded = self.encoding_layer(features_embedded.unsqueeze(2))
        labels_encoded = self.encoding_layer(labels_embedded.unsqueeze(2))
        query_encoded = self.encoding_layer(query_embedded.unsqueeze(1).unsqueeze(2))

        mem, write_key, write_val = self.writing_layer(torch.cat((features_encoded, labels_encoded), dim=-1))

        read_key, read_val = self.reading_layer(query_encoded, mem)

        outputs = torch.sum(read_val[:, -30:, :], dim=1)
        outputs = self.output_layer(outputs)

        encoding_outputs = [features_encoded, labels_encoded, query_encoded]
        writing_outputs = [mem, write_key, write_val]
        reading_outputs = [read_key, read_val]

        return outputs, encoding_outputs, writing_outputs, reading_outputs

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.input_layer.weight.data, gain=math.sqrt(2))
