"""Wrapper over `torch.utils.data.Sampler` for distributed training.

Taken from https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
"""

from operator import itemgetter
from typing import Optional, Iterator, Any

import torch.utils.data
import torch.utils.data.distributed


class DatasetFromSampler(torch.utils.data.Dataset):

    def __init__(self, sampler: torch.utils.data.sampler) -> None:
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int) -> Any:
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)

        return self.sampler_list[index]

    def __len__(self) -> int:
        return len(self.sampler)


class DistributedSamplerWrapper(torch.utils.data.distributed.DistributedSampler):

    def __init__(self, sampler: torch.utils.data.sampler, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True) -> None:
        super().__init__(DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        sub_sampler_indexes = self.dataset

        return iter(itemgetter(*indexes_of_indexes)(sub_sampler_indexes))
