"""Episodic batch sampler"""

from typing import List, Generator, Union

import numpy as np
import torch.utils.data
from matplotlib import pyplot as plt

from data.omniglot_dataset import OmniglotDataset


class EpisodicBatchSampler(torch.utils.data.Sampler):

    def __init__(self, labels: List[int], num_classes: int, batch_size: int, iterations: int,
                 seed: Union[None, int]) -> None:
        super().__init__(data_source=None)
        self.labels = labels
        self.num_classes = num_classes
        self.batch_size = batch_size
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

        perm = None
        for it in range(self.iterations):
            batch = []
            for b in range(self.batch_size):
                c_idc = torch.randperm(len(self.classes), generator=self.rng)[:self.num_classes]
                t_idx = c_idc[torch.randperm(len(c_idc), generator=self.rng)[0]]

                for i, idx in enumerate(c_idc):
                    if idx == t_idx:
                        perm = torch.randperm(self.num_elements_per_class[idx], generator=self.rng)
                        sample_idc = perm[0]
                    else:
                        sample_idc = torch.randperm(self.num_elements_per_class[idx], generator=self.rng)[0]

                    batch.append(self.indices[idx][sample_idc])

                sample_idc = perm[1]
                batch.append(self.indices[t_idx][sample_idc])

            yield torch.stack(batch).long()

    def __len__(self) -> int:
        return self.iterations


def main():
    batch_size = 2
    iterations = 2
    num_classes = 5
    val_set = OmniglotDataset(mode='val', root='../data')
    val_sampler = EpisodicBatchSampler(val_set.y, num_classes=num_classes, batch_size=batch_size, iterations=iterations,
                                       seed=None)

    for i, item in enumerate(val_sampler):
        print(i, item)

    val_loader = torch.utils.data.DataLoader(val_set, batch_sampler=val_sampler)

    for i, sample in enumerate(val_loader):
        x, y = sample
        x = x.view(batch_size, -1, *x.size()[1:])
        y = y.view(batch_size, -1)
        print(x.size(), y)

        facts = x[:, :num_classes]

        labels = torch.arange(0, num_classes)
        labels = labels.expand(batch_size, num_classes).long()

        query = x[:, -1]

        answer = labels[y[:, :-1] == y[:, -1].view(batch_size, -1)]

        print(facts.size(), labels, query.size(), answer)

        fig, ax = plt.subplots(nrows=2, ncols=val_sampler.num_classes)
        for j in range(val_sampler.num_classes):
            img = facts[0][j].numpy()
            ax[0, j].imshow(np.transpose(img, (1, 2, 0)))
        img = query[0].numpy()
        ax[1, -1].imshow(np.transpose(img, (1, 2, 0)))

        fig, ax = plt.subplots(nrows=2, ncols=val_sampler.num_classes)
        for j in range(val_sampler.num_classes):
            img = facts[1][j].numpy()
            ax[0, j].imshow(np.transpose(img, (1, 2, 0)))
        img = query[1].numpy()
        ax[1, -1].imshow(np.transpose(img, (1, 2, 0)))
        plt.show()


if __name__ == '__main__':
    main()
