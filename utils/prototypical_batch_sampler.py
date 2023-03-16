"""Prototypical batch sampler"""

from typing import List, Generator, Union

import numpy as np
import torch
import torch.utils.data


class PrototypicalBatchSampler(torch.utils.data.Sampler):
    """PrototypicalBatchSampler: yield a batch of indexes at each iteration.

    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.
    """

    def __init__(self, labels: List[int], classes_per_iteration: int, samples_per_class: int, iterations: int,
                 seed: Union[None, int]) -> None:
        super().__init__(data_source=None)
        self.labels = labels
        self.classes_per_iteration = classes_per_iteration
        self.samples_per_class = samples_per_class
        self.iterations = iterations
        self.seed = seed

        if self.seed is not None:
            self.rng = torch.Generator()
            self.rng = self.rng.manual_seed(self.seed)
        else:
            self.rng = None

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        self.idc = range(len(self.labels))
        self.indices = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indices = torch.Tensor(self.indices)
        self.num_elements_per_class = torch.zeros_like(self.classes)

        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indices[label_idx, np.where(np.isnan(self.indices[label_idx]))[0][0]] = idx
            self.num_elements_per_class[label_idx] += 1

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        for it in range(self.iterations):
            batch_size = self.samples_per_class * self.classes_per_iteration
            batch = torch.LongTensor(batch_size)
            c_idc = torch.randperm(len(self.classes), generator=self.rng)[:self.classes_per_iteration]
            for i, c in enumerate(self.classes[c_idc]):
                s = slice(i * self.samples_per_class, (i + 1) * self.samples_per_class)
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idc = torch.randperm(self.num_elements_per_class[label_idx], generator=self.rng)[
                             :self.samples_per_class]
                batch[s] = self.indices[label_idx][sample_idc]
            batch = batch[torch.randperm(len(batch), generator=self.rng)]

            yield batch

    def __len__(self) -> int:
        return self.iterations
