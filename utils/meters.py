"""Utilities for training and evaluation meters"""

from typing import List


class ProgressMeter(object):
    def __init__(self, num_batches: int, meters: List[object], prefix: str = "") -> None:
        self.batch_format_string = self._get_batch_format_string(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_format_string.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_format_string(num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'

        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ':f') -> None:
        self.name = name
        self.fmt = fmt

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        format_string = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'

        return format_string.format(**self.__dict__)
