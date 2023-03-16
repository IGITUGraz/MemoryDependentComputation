import argparse
import os
import random
import socket
import warnings
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.utils.data
from torch.backends import cudnn
from tqdm import tqdm

from data.omniglot_dataset import OmniglotDataset
from functions.loss_functions import prototypical_loss as loss_fn
from models.protonet_models import ProtoNet
from utils import checkpoint
from utils.prototypical_batch_sampler import PrototypicalBatchSampler

parser = argparse.ArgumentParser(description='Protonet Omniglot training')
parser.add_argument('--dir', default='../data', type=str, metavar='DIR',
                    help='Path to dataset (default: ../data)')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='Number of total epochs to run (default: 100)')
parser.add_argument('--iterations', default=100, type=int, metavar='N',
                    help='Number of episodes per epoch (default: 100)')

parser.add_argument('--learning_rate', default=0.001, type=float, metavar='N',
                    help='Initial learning rate (default: 0.001)')
parser.add_argument('--lr_scheduler_step', default=20, type=int, metavar='N',
                    help='Step parameter for StepLR learning rate scheduler (default: 20)')
parser.add_argument('--lr_scheduler_gamma', default=0.5, type=float, metavar='N',
                    help='Gamma parameter for StepLR learning rate scheduler (default: 0.5)')

parser.add_argument('--classes_per_iteration_train', default=60, type=int, metavar='N',
                    help='Number of random classes per episode for training (default: 60)')
parser.add_argument('--num_samples_support_train', default=1, type=int, metavar='N',
                    help='Number of samples per class to use as support for training (default: 1 for 1-shot)')
parser.add_argument('--num_samples_query_train', default=5, type=int, metavar='N',
                    help='Number of samples per class to use as query for training (default: 5)')

parser.add_argument('--classes_per_iteration_val', default=5, type=int, metavar='N',
                    help='Number of random classes per episode for validation (default: 5 for 5-way)')
parser.add_argument('--num_samples_support_val', default=1, type=int, metavar='N',
                    help='Number of samples per class to use as support for validation (default: 1 for 1-shot)')
parser.add_argument('--num_samples_query_val', default=15, type=int, metavar='N',
                    help='Number of samples per class to use as query for validation (default: 15)')

parser.add_argument('--use_batch_norm', action='store_true',
                    help='Use batch-norm layers in the ProtoNet')
parser.add_argument('--evaluate', action='store_true',
                    help='Evaluate the model on the test set')
parser.add_argument('--checkpoint_path', default='', type=str, metavar='PATH',
                    help='Path to latest checkpoint (default: none)')

parser.add_argument('--seed', default=7, type=int, metavar='N',
                    help='Seed for initializing training (default: 7)')
parser.add_argument('--sampler_seed', default=7, type=int, metavar='N',
                    help='Seed for prototypical batch sampler (default: 7)')
parser.add_argument('--cuda', action='store_true',
                    help='Enables CUDA')


def init_seed(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def init_dataset(mode: str, args: argparse.Namespace) -> OmniglotDataset:
    dataset = OmniglotDataset(mode=mode, root=args.dir)
    n_classes = len(np.unique(dataset.y))
    if n_classes < args.classes_per_iteration_train or n_classes < args.classes_per_iteration_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset


def init_sampler(mode: str, labels: List[int], args: argparse.Namespace) -> PrototypicalBatchSampler:
    if 'train' in mode:
        classes_per_it = args.classes_per_iteration_train
        num_samples = args.num_samples_support_train + args.num_samples_query_train
    else:
        classes_per_it = args.classes_per_iteration_val
        num_samples = args.num_samples_support_val + args.num_samples_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_iteration=classes_per_it,
                                    samples_per_class=num_samples,
                                    iterations=args.iterations,
                                    seed=args.sampler_seed)


def init_dataloader(mode: str, args: argparse.Namespace) -> torch.utils.data.DataLoader:
    dataset = init_dataset(mode, args)
    sampler = init_sampler(mode, dataset.y, args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

    return dataloader


def init_protonet(args: argparse.Namespace) -> ProtoNet:
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'
    model = ProtoNet(use_batch_norm=args.use_batchnorm).to(device)

    return model


def init_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Adam:
    return torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)


def init_lr_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> torch.optim.lr_scheduler.StepLR:
    return torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                           gamma=args.lr_scheduler_gamma,
                                           step_size=args.lr_scheduler_step)


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler,
          tr_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader,
          args: argparse.Namespace) -> Tuple[Optional[Dict[str, torch.Tensor]], float, list, list, list, list]:
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'

    with open('./version.txt') as f:
        version = f.readline()

    time_stamp = datetime.now().strftime('%b%d_%H-%M-%S')

    if val_dataloader is None:
        best_state = None
    else:
        best_state = model.state_dict()

    best_acc = 0.0
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    for epoch in range(args.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optimizer.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y, num_support=args.num_samples_support_train)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())

        avg_loss = np.mean(train_loss[-args.iterations:])
        avg_acc = np.mean(train_acc[-args.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))

        lr_scheduler.step()
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)

        model.eval()
        for batch in val_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y, num_support=args.num_samples_support_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())

        avg_loss = np.mean(val_loss[-args.iterations:])
        avg_acc = np.mean(val_acc[-args.iterations:])
        is_best = avg_acc >= best_acc
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(avg_loss, avg_acc, postfix))

        if is_best:
            best_acc = avg_acc
            best_state = model.state_dict()

        checkpoint.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'time_stamp': time_stamp,
            'params': args
        }, is_best, filename=os.path.join(
            './results', 'checkpoints', time_stamp +
            '_' + socket.gethostname() +
            f'_version-{version}_protonet_omniglot-{args.num_samples_support_train}'
            f'-shot-{args.classes_per_iteration_val}-way'))

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(model: torch.nn.Module, test_dataloader: torch.utils.data.DataLoader, args: argparse.Namespace) -> np.ndarray:
    device = 'cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu'

    avg_acc = []
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            _, acc = loss_fn(model_output, target=y, num_support=args.num_samples_support_val)
            avg_acc.append(acc.item())

    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc


def main():
    args = parser.parse_args()

    if not os.path.exists('./results'):
        os.makedirs('./results')

    if torch.cuda.is_available() and not args.cuda:
        warnings.warn("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(args)

    tr_dataloader = init_dataloader('train', args)
    val_dataloader = init_dataloader('val', args)
    test_dataloader = init_dataloader('test', args)

    model = init_protonet(args)
    optimizer = init_optimizer(model, args)
    lr_scheduler = init_lr_scheduler(optimizer, args)

    if not args.evaluate:
        res = train(model=model, optimizer=optimizer, tr_dataloader=tr_dataloader, val_dataloader=val_dataloader,
                    lr_scheduler=lr_scheduler, args=args)

        best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
        print('Testing with last model..')
        test(model=model, test_dataloader=test_dataloader, args=args)

        model.load_state_dict(best_state)
        print('Testing with best model..')
        test(model=model, test_dataloader=test_dataloader, args=args)
    else:
        if os.path.isfile(args.checkpoint_path):
            print("=> loading checkpoint '{}'".format(args.checkpoint_path))
            if torch.cuda.is_available() and args.cuda:
                ckp = checkpoint.load_checkpoint(args.checkpoint_path, 'cuda:0')
            else:
                ckp = checkpoint.load_checkpoint(args.checkpoint_path)

            model.load_state_dict(ckp['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint_path, ckp['epoch']))
        elif args.checkpoint_path:
            print("=> no checkpoint found at '{}'".format(args.checkpoint_path))

        print('Evaluating on test set..')
        test(model=model, test_dataloader=test_dataloader, args=args)


if __name__ == '__main__':
    main()
