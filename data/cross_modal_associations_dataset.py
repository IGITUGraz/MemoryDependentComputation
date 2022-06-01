"""Cross-modal associations dataset"""

import glob
import os
import shutil
import subprocess
from typing import Tuple, Callable, Optional, List

import numpy as np
import torch.utils.data
import torchaudio
from numpy.random import Generator
from numpy.random import default_rng
from torchaudio.transforms import MFCC
from torchvision.datasets import MNIST

from functions.utility_functions import TrimSilence


class PostprocessMFCC(object):

    def __init__(self, num_mfcc: int, num_time_samples: int) -> None:
        self.num_mfcc = num_mfcc
        self.num_time_samples = num_time_samples

    def __call__(self, mfcc: torch.Tensor) -> torch.Tensor:
        img = torch.zeros((self.num_mfcc, self.num_time_samples)).type(torch.FloatTensor)
        mfcc_sum = torch.sum(mfcc, dim=0)
        max_e = torch.argmax(mfcc_sum)
        if max_e > mfcc.size()[1] - self.num_time_samples // 2:
            max_e = mfcc.size()[1] - self.num_time_samples // 2 + 1
        if max_e < self.num_time_samples // 2:
            max_e = self.num_time_samples // 2
        if mfcc.size()[1] > self.num_time_samples:
            mfcc = mfcc[:, max_e - self.num_time_samples // 2:max_e + self.num_time_samples // 2]
            img[:, :mfcc.size()[1]] = mfcc
        else:
            img[:, :mfcc.size()[1]] = mfcc

        return img


class FSDDataset(torch.utils.data.Dataset):
    """Free spoken digit dataset."""

    version = 'v1.0.10'
    name = 'free-spoken-digit-dataset'
    url = 'https://github.com/Jakobovski/free-spoken-digit-dataset'
    sample_rate = 8000
    num_recordings_per_person_and_digit = 50

    def __init__(self, root: str, train: bool, num_mfcc: int, num_mfcc_time_samples: int,
                 download: Optional[bool] = True, transform: Optional[Callable] = None) -> None:
        self.root = root
        self.train = train
        self.num_mfcc = num_mfcc
        self.num_mfcc_time_samples = num_mfcc_time_samples
        self.transform = transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.file_paths = self._train_test_split()
        self.data, self.targets = self._load()

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple:
        x = self.data[idx]
        y = self.targets[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def download(self) -> None:
        if self._check_exists():
            return

        repo_path = os.path.join(self.root, self.name)
        os.makedirs(repo_path)
        try:
            subprocess.call(['git', '-C', self.root, 'clone', self.url, '--branch', self.version])
            shutil.move(os.path.join(repo_path, 'recordings'), os.path.join(self.root, self.version, 'recordings'))
        finally:
            if os.path.isdir(repo_path):
                shutil.rmtree(repo_path)

    def _check_exists(self) -> bool:
        return os.path.isdir(os.path.join(self.root, self.version, 'recordings'))

    def _train_test_split(self, test_size=0.1) -> List[str]:
        assert 0. < test_size < 1.

        all_file_paths = glob.glob(os.path.join(self.root, self.version, 'recordings', '*.wav'))

        train_file_paths, test_file_paths = [], []
        n_test = int(self.num_recordings_per_person_and_digit * test_size)

        for file in all_file_paths:
            file_name, ext = os.path.splitext(os.path.basename(file))
            digit, name, rec_num = file_name.split('_')
            split = test_file_paths if int(rec_num) + 1 <= n_test else train_file_paths
            split.append(file)

        if self.train:
            return train_file_paths
        else:
            return test_file_paths

    def _load(self) -> Tuple:
        recordings, labels = [], []
        for file_path in self.file_paths:
            audio = torchaudio.load(file_path)[0].flatten() # noqa

            # Trim silence from the start and end of the audio
            audio = TrimSilence(threshold=1e-6)(audio)

            # Generate num_mfcc+1 MFCCs (and remove the first one since it is (almost) a constant offset)
            mfcc = MFCC(sample_rate=self.sample_rate, n_mfcc=self.num_mfcc+1)(audio)[1:, :]

            # Standardize MFCCs for each frame
            mfcc = (mfcc - mfcc.mean(axis=0)) / mfcc.std(axis=0)

            # Create MFCC image
            mfcc = PostprocessMFCC(self.num_mfcc, self.num_mfcc_time_samples)(mfcc)

            recordings.append(mfcc)
            labels.append(int(os.path.basename(file_path)[0]))

        return recordings, labels


class CrossModalAssociationsDataset(torch.utils.data.Dataset):
    """Cross-modal associations dataset."""

    audio_sample_rate = FSDDataset.sample_rate

    def __init__(self, root: str, train: bool, dataset_size: int, sequence_length: int,
                 num_mfcc: int, num_mfcc_time_samples: int, classes: Optional[List[int]] = None,
                 image_transform: Optional[Callable] = None, audio_transform: Optional[Callable] = None,
                 rng: Generator = default_rng()) -> None:
        super().__init__()
        self.train = train
        self.dataset_size = dataset_size
        self.sequence_length = sequence_length
        self.classes = list(range(10)) if classes is None else classes
        self.image_transform = image_transform
        self.audio_transform = audio_transform
        self.rng = rng

        assert sequence_length <= 10

        self.audio_data = FSDDataset(root=os.path.join(root, 'FSDDataset'),
                                     train=train,
                                     num_mfcc=num_mfcc,
                                     num_mfcc_time_samples=num_mfcc_time_samples,
                                     download=True,
                                     transform=audio_transform)
        self.image_data = MNIST(os.path.join(root, 'MNISTDataset'), train, download=True, transform=image_transform)

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, idx: int) -> Tuple:
        audio_classes = self.rng.choice(self.classes, self.sequence_length, replace=False)
        target_class = self.rng.choice(audio_classes)

        if self.train:
            image_classes = self.rng.permutation(audio_classes)
        else:
            image_classes = audio_classes

        mfcc_class_sequence, image_class_sequence, mfcc_query_class, image_target_class = [], [], None, None
        mfcc_sequence, image_sequence, mfcc_query, image_target = [], [], None, None
        for ac, ic in zip(audio_classes, image_classes):
            if ac == target_class:
                audio_idc = self.rng.choice(np.where(self.audio_data.targets == ac)[0], 2, replace=False)
                audio_idx, query_idx = audio_idc[0], audio_idc[1]
                image_idx = self.rng.choice(np.where(self.image_data.targets == ic)[0])
                mfcc_query = self.audio_data[query_idx][0]
                mfcc_query_class = self.audio_data[query_idx][1]
                image_target = self.image_data[image_idx][0]
                image_target_class = self.image_data[image_idx][1]
            else:
                audio_idx = self.rng.choice(np.where(self.audio_data.targets == ac)[0])
                image_idx = self.rng.choice(np.where(self.image_data.targets == ic)[0])

            mfcc_sequence.append(self.audio_data[audio_idx][0])
            image_sequence.append(self.image_data[image_idx][0])
            mfcc_class_sequence.append(self.audio_data[audio_idx][1])
            image_class_sequence.append(self.image_data[image_idx][1])

        targets = (mfcc_class_sequence, image_class_sequence, mfcc_query_class, image_target_class)

        return torch.stack(mfcc_sequence, dim=0), torch.stack(image_sequence, dim=0), mfcc_query, image_target, targets
