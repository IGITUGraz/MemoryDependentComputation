"""Omniglot dataset"""

import errno
import os
import shutil
import zipfile
from typing import Tuple, Any, Optional, Callable, List, Dict

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from six.moves import urllib

IMG_CACHE = {}


class OmniglotDataset(torch.utils.data.Dataset):
    """Omniglot dataset"""

    vinyals_split_url = \
        'https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/'

    vinyals_split_sizes = {
        'test': vinyals_split_url + 'test.txt',
        'train': vinyals_split_url + 'train.txt',
        'trainval': vinyals_split_url + 'trainval.txt',
        'val': vinyals_split_url + 'val.txt',
    }

    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]

    splits_folder = os.path.join('splits', 'vinyals')
    raw_folder = 'raw'
    processed_folder = 'data'

    def __init__(self, root: str, mode: str = 'train', download: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it.')

        if os.path.exists(os.path.join(self.folder, f'data-{mode}.pt')):
            data = torch.load(os.path.join(self.folder, f'data-{mode}.pt'))
            self.x = data['x']
            self.y = data['y']
        else:
            self.classes = self._get_current_classes(mode)
            self.all_items = self._find_items(self.classes)

            self.idx_classes = self._index_classes(self.all_items)

            paths, self.y = zip(*[self._get_path_label(pl) for pl in range(len(self))])

            self.x = map(self._load_img, paths)
            self.x = list(self.x)

            torch.save({'x': self.x, 'y': self.y}, os.path.join(self.folder, f'data-{mode}.pt'))

    def __len__(self) -> int:
        return len(self.all_items)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        x = self.x[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y

    @property
    def folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)

    @staticmethod
    def _index_classes(items: List[Tuple[str, str, str, str]]) -> Dict[str, int]:
        idx = {}
        for i in items:
            if not i[1] + i[-1] in idx:
                idx[i[1] + i[-1]] = len(idx)
        print("== Dataset: Found %d classes" % len(idx))

        return idx

    @staticmethod
    def _load_img(path: str) -> torch.Tensor:
        path, rot = path.split(os.sep + 'rot')
        if path in IMG_CACHE:
            x = IMG_CACHE[path]
        else:
            x = Image.open(path)
            IMG_CACHE[path] = x
        x = x.rotate(float(rot))
        x = x.resize((28, 28))

        shape = 1, x.size[0], x.size[1]
        x = np.array(x, np.float32, copy=False)  # noqa
        x = 1.0 - torch.from_numpy(x)
        x = x.transpose(0, 1).view(shape)

        return x

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.folder, self.processed_folder))

    def _get_current_classes(self, mode: str) -> List[str]:
        file_name = os.path.join(self.folder, self.splits_folder, mode + '.txt')

        with open(file_name) as f:
            classes = f.read().replace('/', os.sep).splitlines()

        return classes

    def _find_items(self, classes: List[str]) -> List[Tuple[str, str, str, str]]:
        root_dir = os.path.join(self.folder, self.processed_folder)

        items = []
        rots = [os.sep + 'rot000', os.sep + 'rot090', os.sep + 'rot180', os.sep + 'rot270']
        for (root, dirs, files) in os.walk(root_dir):
            for f in files:
                r = root.split(os.sep)
                lr = len(r)
                label = r[lr - 2] + os.sep + r[lr - 1]
                for rot in rots:
                    if label + rot in classes and (f.endswith("png")):
                        items.extend([(f, label, root, rot)])
        print("== Dataset: Found %d items " % len(items))

        return items

    def _get_path_label(self, index: int) -> Tuple[str, int]:
        filename = self.all_items[index][0]
        rot = self.all_items[index][-1]
        img = str.join(os.sep, [self.all_items[index][2], filename]) + rot
        target = self.idx_classes[self.all_items[index][1] + self.all_items[index][-1]]

        return img, target

    def download(self) -> None:

        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.folder, self.splits_folder))
            os.makedirs(os.path.join(self.folder, self.raw_folder))
            os.makedirs(os.path.join(self.folder, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for k, url in self.vinyals_split_sizes.items():
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition(os.sep)[-1]
            file_path = os.path.join(self.folder, self.splits_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())

        orig_root = ''
        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition(os.sep)[2]
            file_path = os.path.join(self.folder, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            orig_root = os.path.join(self.folder, self.raw_folder)
            print("== Unzip from " + file_path + " to " + orig_root)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(orig_root)
            zip_ref.close()

        file_processed = os.path.join(self.folder, self.processed_folder)
        for p in ['images_background', 'images_evaluation']:
            for f in os.listdir(os.path.join(orig_root, p)):
                shutil.move(os.path.join(orig_root, p, f), file_processed)
            os.rmdir(os.path.join(orig_root, p))
        print("Download finished.")
