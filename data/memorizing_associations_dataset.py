"""Memorizing associations dataset."""

from typing import Tuple, TypedDict, Any

import numpy as np
import torch.utils.data
from numpy.random import default_rng


class MemorizingAssociationsDataset(torch.utils.data.Dataset):
    """Memorizing associations dataset"""

    def __init__(self, sequence_length: int, num_classes: int, feature_size: int, dataset_size: int, inf_data: bool,
                 seed: int) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.dataset_size = dataset_size
        self.inf_data = inf_data
        self.seed = seed

        self.rng = default_rng(self.seed)

        if not inf_data:
            self.data = self._create_dataset()

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> Tuple[TypedDict('MemorizingAssociationsDatasetItem',
                                                       {'features': Any,'labels': Any, 'query': Any, 'answer': Any}),
                                             int]:
        if not self.inf_data:
            features = self.data[0][idx]
            labels = self.data[1][idx]
            query = self.data[2][idx]
            answer = self.data[3][idx]
        else:
            features = self.rng.random((self.sequence_length, self.feature_size), dtype=np.float64)
            labels = self.rng.choice(self.num_classes, self.sequence_length, replace=False)

            idx = self.rng.choice(self.sequence_length)
            query = features[idx]
            answer = labels[idx]

        length = self.sequence_length

        return {'features': torch.from_numpy(features).float(), 'labels': torch.from_numpy(labels),
                'query': torch.from_numpy(query).float(), 'answer': answer}, length

    def _create_dataset(self) -> Tuple:
        features, labels, queries, answers = [], [], [], []
        for i in range(self.dataset_size):
            f = self.rng.random((self.sequence_length, self.feature_size), dtype=np.float64)
            c = self.rng.choice(self.num_classes, self.sequence_length, replace=False)

            idx = self.rng.choice(self.sequence_length)
            q = f[idx]
            a = c[idx]

            features.append(f)
            labels.append(c)
            queries.append(q)
            answers.append(a)

        return features, labels, queries, answers

    def set_worker_seed(self, worker_id):
        self.rng = default_rng(self.seed + worker_id)
