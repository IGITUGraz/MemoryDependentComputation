"""Network models"""

import math
from typing import Tuple, Callable, List

import torch
import torch.nn.functional

from layers.dense import DenseLayer
from layers.embedding import EmbeddingLayer
from layers.encoding import EncodingLayer
from layers.reading import ReadingLayer
from layers.writing import WritingLayer
from models.neuron_models import NeuronModel
from models.spiking_protonet import SpikingProtoNet
from policies import policy


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

        self.reset_parameters()

    def forward(self, features: torch.Tensor, labels: torch.Tensor, query: torch.Tensor) -> Tuple:
        features_embedded = self.input_layer(features)
        labels_embedded = self.embedding_layer(labels)
        query_embedded = self.input_layer(query)

        features_encoded, _ = self.encoding_layer(features_embedded.unsqueeze(2))
        labels_encoded, _ = self.encoding_layer(labels_embedded.unsqueeze(2))
        query_encoded, _ = self.encoding_layer(query_embedded.unsqueeze(1).unsqueeze(2))

        mem, write_key, write_val, _ = self.writing_layer(torch.cat((features_encoded, labels_encoded), dim=-1))

        read_key, read_val, _ = self.reading_layer(query_encoded, mem)

        outputs = torch.sum(read_val[:, -30:, :], dim=1)
        outputs = self.output_layer(outputs)

        encoding_outputs = [features_encoded, labels_encoded, query_encoded]
        writing_outputs = [mem, write_key, write_val]
        reading_outputs = [read_key, read_val]

        return outputs, encoding_outputs, writing_outputs, reading_outputs

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.input_layer.weight.data, gain=math.sqrt(2))


class OmniglotOneShot(torch.nn.Module):

    def __init__(self, num_embeddings: int, output_size: int, memory_size: int, num_time_steps: int, readout_delay: int,
                 tau_trace: float, image_embedding_layer: SpikingProtoNet, plasticity_rule: Callable,
                 dynamics: NeuronModel) -> None:
        super().__init__()
        spiking_image_size = image_embedding_layer.input_size
        embedding_size = image_embedding_layer.output_size

        self.image_encoding_layer = EncodingLayer(1, spiking_image_size, False, False, num_time_steps, dynamics)
        self.encoding_layer = EncodingLayer(1, embedding_size, False, False, num_time_steps, dynamics)
        self.images_embedding_layer = image_embedding_layer
        self.labels_embedding_layer = EmbeddingLayer(num_embeddings, embedding_size)
        self.writing_layer = WritingLayer(2 * embedding_size, memory_size, plasticity_rule, tau_trace, dynamics)
        self.reading_layer = ReadingLayer(embedding_size, memory_size, readout_delay, dynamics)
        self.output_layer = torch.nn.Linear(memory_size, output_size, bias=False)

    def forward(self, images: torch.Tensor, labels: torch.Tensor, query: torch.Tensor) -> Tuple:
        batch_size, sequence_length, *CHW = images.size()

        images_encoded_sequence = []
        for t in range(sequence_length):
            images_spiking, _ = self.image_encoding_layer(torch.flatten(images.select(1, t), -2, -1).unsqueeze(2))
            images_embedded = self.images_embedding_layer(images_spiking)
            images_encoded_sequence.append(images_embedded)
        images_encoded = torch.cat(images_encoded_sequence, dim=1)

        query_spiking, _ = self.image_encoding_layer(torch.flatten(query, -2, -1).unsqueeze(2))
        query_encoded = self.images_embedding_layer(query_spiking)

        labels_embedded = self.labels_embedding_layer(labels)
        labels_encoded, _ = self.encoding_layer(labels_embedded.unsqueeze(2))

        mem, write_key, write_val, _ = self.writing_layer(torch.cat((images_encoded, labels_encoded), dim=-1))

        read_key, read_val, _ = self.reading_layer(query_encoded, mem)

        outputs = torch.sum(read_val[:, -30:, :], dim=1)
        outputs = self.output_layer(outputs)

        encoding_outputs = [images_encoded, labels_encoded, query_encoded]
        writing_outputs = [mem, write_key, write_val]
        reading_outputs = [read_key, read_val]

        return outputs, encoding_outputs, writing_outputs, reading_outputs


class CrossModalAssociations(torch.nn.Module):

    def __init__(self, output_size: int, memory_size: int, num_time_steps: int, readout_delay: int, tau_trace: float,
                 mfcc_embedding_layer: torch.nn.Module, image_embedding_layer: torch.nn.Module,
                 plasticity_rule: Callable, dynamics: NeuronModel) -> None:
        super().__init__()
        spiking_mfcc_size = mfcc_embedding_layer.input_size
        spiking_image_size = image_embedding_layer.input_size
        mfcc_feature_size = mfcc_embedding_layer.output_size
        image_feature_size = image_embedding_layer.output_size
        writing_layer_input_size = mfcc_feature_size + image_feature_size

        assert mfcc_feature_size == image_feature_size

        self.image_encoding_layer = EncodingLayer(1, spiking_image_size, False, False, num_time_steps, dynamics)
        self.mfcc_encoding_layer = EncodingLayer(1, spiking_mfcc_size, False, False, num_time_steps, dynamics)
        self.mfcc_embedding_layer = mfcc_embedding_layer
        self.image_embedding_layer = image_embedding_layer
        self.writing_layer = WritingLayer(writing_layer_input_size, memory_size, plasticity_rule, tau_trace, dynamics)
        self.reading_layer = ReadingLayer(mfcc_feature_size, memory_size, readout_delay, dynamics)

        self.decoder_l1 = DenseLayer(memory_size, 256, dynamics)
        self.decoder_l2 = DenseLayer(256, output_size, dynamics)

    def forward(self, mfcc: torch.Tensor, images: torch.Tensor, query: torch.Tensor) -> Tuple:
        batch_size, sequence_length, *CHW = mfcc.size()

        mfcc_encoded_sequence, images_encoded_sequence = [], []
        for t in range(sequence_length):
            images_spiking, _ = self.image_encoding_layer(torch.flatten(images.select(1, t), -2, -1).unsqueeze(2))
            mfcc_spiking, _ = self.mfcc_encoding_layer(torch.flatten(mfcc.select(1, t), -2, -1).unsqueeze(2))
            images_embedded = self.image_embedding_layer(images_spiking)
            mfcc_embedded = self.mfcc_embedding_layer(mfcc_spiking)
            images_encoded_sequence.append(images_embedded)
            mfcc_encoded_sequence.append(mfcc_embedded)
        images_encoded = torch.cat(images_encoded_sequence, dim=1)
        mfcc_encoded = torch.cat(mfcc_encoded_sequence, dim=1)

        query_spiking, _ = self.mfcc_encoding_layer(torch.flatten(query, -2, -1).unsqueeze(2))
        query_encoded = self.mfcc_embedding_layer(query_spiking)

        mem, write_key, write_val, _ = self.writing_layer(torch.cat((mfcc_encoded, images_encoded), dim=-1))

        read_key, read_val, _ = self.reading_layer(query_encoded, mem)

        decoder_output_l1, _, _ = self.decoder_l1(read_val)
        decoder_output_l2, _, _ = self.decoder_l2(decoder_output_l1)

        outputs = torch.sum(decoder_output_l2, dim=1).squeeze() / 15

        encoding_outputs = [mfcc_encoded, images_encoded, query_encoded]
        writing_outputs = [mem, write_key, write_val]
        reading_outputs = [read_key, read_val]
        decoder_outputs = [decoder_output_l1, decoder_output_l2]

        return outputs, encoding_outputs, writing_outputs, reading_outputs, decoder_outputs


class QuestionAnswering(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int, num_embeddings: int, embedding_size: int, memory_size: int,
                 mask_time_words: bool, learn_encoding: bool, num_time_steps: int, readout_delay: int, tau_trace: float,
                 plasticity_rule: Callable, dynamics: NeuronModel) -> None:
        super().__init__()
        self.readout_delay = readout_delay

        self.embedding_layer = EmbeddingLayer(num_embeddings, embedding_size, padding_idx=0)
        self.encoding_layer = EncodingLayer(input_size, embedding_size, mask_time_words, learn_encoding,
                                            num_time_steps, dynamics)
        self.writing_layer = WritingLayer(embedding_size, memory_size, plasticity_rule, tau_trace, dynamics)
        self.reading_layer = ReadingLayer(embedding_size, memory_size, readout_delay, dynamics)
        self.output_layer = torch.nn.Linear(memory_size, output_size, bias=False)

    def forward(self, story: torch.Tensor, query: torch.Tensor) -> Tuple:

        story_embedded = self.embedding_layer(story)
        query_embedded = self.embedding_layer(query)

        story_encoded, _ = self.encoding_layer(story_embedded)
        query_encoded, _ = self.encoding_layer(query_embedded.unsqueeze(1))

        mem, write_key, write_val, _ = self.writing_layer(story_encoded)

        read_key, read_val, _ = self.reading_layer(query_encoded, mem)

        outputs = torch.sum(read_val[:, -30:, :], dim=1)
        outputs = self.output_layer(outputs)

        encoding_outputs = [story_encoded, query_encoded]
        writing_outputs = [mem, write_key, write_val]
        reading_outputs = [read_key, read_val]

        return outputs, encoding_outputs, writing_outputs, reading_outputs


class ReinforcementLearningBase(torch.nn.Module):

    def __init__(self, input_size: int, embedding_size: int, memory_size: int, num_time_steps: int, readout_delay: int,
                 tau_trace: float, plasticity_rule: Callable, dynamics: NeuronModel) -> None:
        super().__init__()
        self.memory_size = memory_size
        self.num_time_steps = num_time_steps

        self.embedding_layer = torch.nn.Linear(input_size, embedding_size, bias=False)
        self.encoding_layer = EncodingLayer(embedding_size, embedding_size, False, False, num_time_steps, dynamics)
        self.writing_layer = WritingLayer(embedding_size, memory_size, plasticity_rule, tau_trace, dynamics)
        self.reading_layer = ReadingLayer(embedding_size, memory_size, readout_delay, dynamics)

        self.reset_parameters()

    @property
    def state_size(self):
        return (self.memory_size, self.memory_size),\
               (80,), (80,), (80,),\
               (self.memory_size,), (self.memory_size,), (self.memory_size,), (self.memory_size,), (self.memory_size,),\
               (self.memory_size,), (self.memory_size,), (self.memory_size,),\
               (self.memory_size,), (self.memory_size,), (self.memory_size,), (self.memory_size,), (self.memory_size,),\
               (self.memory_size,), (1, self.memory_size),

    @property
    def output_size(self) -> int:
        return self.memory_size

    def forward(self, facts: torch.Tensor, states: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        facts = facts.permute(1, 0, -1)
        batch_size, sequence_length, _ = facts.size()

        mem = states[0]
        encoding_layer_states = states[1:4]
        writing_layer_states = [states[4:7], states[7:10], states[10], states[11]]
        reading_layer_states = [states[12:15], states[15:18], states[18]]

        output_sequence = []
        for t in range(sequence_length):

            facts_embedded = self.embedding_layer(facts.select(1, t)).unsqueeze(1).unsqueeze(2)

            facts_encoded, encoding_layer_states = self.encoding_layer(facts_embedded, states=encoding_layer_states)

            mem, write_key, write_val, writing_layer_states = self.writing_layer(facts_encoded, mem=mem,
                                                                                 states=writing_layer_states)

            read_key, read_val, reading_layer_states = self.reading_layer(facts_encoded, mem=mem,
                                                                          states=reading_layer_states)

            output_sequence.append(torch.sum(read_val[:, -30:, :], dim=1))

        outputs = torch.stack(output_sequence, dim=1)

        states = [mem, *encoding_layer_states]
        for s in writing_layer_states:
            if not isinstance(s, tuple):
                states.append(s)
            else:
                states += s
        for s in reading_layer_states:
            if not isinstance(s, tuple):
                states.append(s)
            else:
                states += s

        return outputs.permute(1, 0, -1), states

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.embedding_layer.weight.data, gain=math.sqrt(2))


class ReinforcementLearning(policy.RNNBase):

    def __init__(self, input_size: int, embedding_size: int, memory_size: int, num_time_steps: int, readout_delay: int,
                 tau_trace: float, plasticity_rule: Callable, dynamics: NeuronModel) -> None:
        super().__init__(
            ReinforcementLearningBase(input_size=input_size,
                                      embedding_size=embedding_size,
                                      memory_size=memory_size,
                                      num_time_steps=num_time_steps,
                                      readout_delay=readout_delay,
                                      tau_trace=tau_trace,
                                      plasticity_rule=plasticity_rule,
                                      dynamics=dynamics)
        )

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(memory_size, memory_size), torch.nn.Tanh(),
            torch.nn.Linear(memory_size, memory_size), torch.nn.Tanh())

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(input_size + memory_size, memory_size), torch.nn.Tanh(),
            torch.nn.Linear(memory_size, memory_size), torch.nn.Tanh())

        self.critic_linear = torch.nn.Linear(memory_size, 1)

        self.reset_parameters()

    @property
    def is_recurrent(self) -> True:
        return True

    def forward(self, inputs: torch.Tensor, states: List[torch.Tensor], masks: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor, List[torch.Tensor]]:

        outputs, states = self._forward(inputs, states, masks)

        actor_outputs = self.actor(outputs)
        critic_outputs = self.critic(torch.cat([inputs, outputs], dim=-1))

        return self.critic_linear(critic_outputs), actor_outputs, states

    def reset_parameters(self) -> None:
        for layer in self.actor:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                torch.nn.init.constant_(layer.bias, val=0.0)

        for layer in self.critic:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                torch.nn.init.constant_(layer.bias, val=0.0)

        torch.nn.init.orthogonal_(self.critic_linear.weight, gain=math.sqrt(2))
        torch.nn.init.constant_(self.critic_linear.bias, val=0.0)
