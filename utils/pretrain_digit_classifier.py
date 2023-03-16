import argparse
import json
import os
import random
import socket
import time
import warnings
from datetime import datetime
from math import ceil, floor

import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Lambda

import utils.checkpoint
import utils.meters
import utils.metrics
from data.cross_modal_associations_dataset import FSDDataset
from models.protonet_models import conv_block

parser = argparse.ArgumentParser(description='MNIST or Audio based digit classification model training')
parser.add_argument('--audio', action='store_true',
                    help='Train digit classifier from the FSD audio dataset')
parser.add_argument('--num_mfcc', default=20, type=int, metavar='NUM_MFCC',
                    help='Number of Mel-frequency cepstrum coefficients (default: 20)')
parser.add_argument('--num_mfcc_time_samples', default=30, type=int, metavar='NUM_MFCC_TIME_SAMPLES',
                    help='Number of time samples for the Mel-frequency cepstrum coefficients (default: 30)')
parser.add_argument('--use_batch_norm', action='store_true',
                    help='Use batch-norm layers in the CNN')

parser.add_argument('--dir', default='../data', type=str, metavar='DIR',
                    help='Path to dataset (default: ../data)')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='Number of data loading workers; set to 0 due to pickling issue (default: 4)')
parser.add_argument('--prefetch_factor', default=2, type=int, metavar='N',
                    help='Prefetch prefetch_factor * workers examples (default: 2)')
parser.add_argument('--pin_data_to_memory', default=1, choices=[0, 1], type=int, metavar='PIN_DATA_TO_MEMORY',
                    help='Pin data to memory (default: 1)')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='Number of total epochs to run (default: 100)')
parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                    help='Mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--learning_rate', default=0.001, type=float, metavar='N',
                    help='Initial learning rate (default: 0.001)')
parser.add_argument('--learning_rate_decay', default=0.85, type=float, metavar='N',
                    help='Learning rate decay (default: 0.85)')
parser.add_argument('--decay_learning_rate_every', default=20, type=int, metavar='N',
                    help='Decay the learning rate every N epochs (default: 20)')

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='Manual epoch number (useful on restarts, default: 0)')
parser.add_argument('--evaluate', action='store_true',
                    help='Evaluate the model on the test set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Path to latest checkpoint (default: none)')

parser.add_argument('--logging', action='store_true',
                    help='Write tensorboard logs')
parser.add_argument('--print_freq', default=100, type=int, metavar='N',
                    help='Print frequency (default: 1)')
parser.add_argument('--world_size', default=-1, type=int, metavar='N',
                    help='Number of nodes for distributed training (default: -1)')
parser.add_argument('--rank', default=-1, type=int, metavar='N',
                    help='Node rank for distributed training (default: -1)')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:23456', type=str, metavar='DIST_URL',
                    help='URL used to set up distributed training (default: tcp://127.0.0.1:23456)')
parser.add_argument('--dist_backend', default='nccl', choices=['nccl', 'mpi', 'gloo'], type=str, metavar='DIST_BACKEND',
                    help='Distributed backend to use (default: nccl)')
parser.add_argument('--seed', default=61, type=int, metavar='N',
                    help='Seed for initializing training (default: none)')
parser.add_argument('--data_seed', default=61, type=int, metavar='N',
                    help='Seed for the dataset (default: none)')
parser.add_argument('--gpu', default=None, type=int, metavar='N',
                    help='GPU id to use (default: none)')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_loss = np.inf
log_dir = ''
writer = None
time_stamp = datetime.now().strftime('%b%d_%H-%M-%S')
suffix = ''

with open('version.txt') as f:
    version = f.readline()


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    num_gpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have num_gpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = num_gpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=num_gpus_per_node, args=(num_gpus_per_node, args))  # noqa
    else:
        # Simply call main_worker function
        main_worker(args.gpu, num_gpus_per_node, args)


class DigitClassifier(torch.nn.Module):

    def __init__(self, input_size: int = 1, hidden_size: int = 64, output_size: int = 64, num_classes: int = 10,
                 use_batch_norm: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoder = torch.nn.Sequential(
            conv_block(input_size, hidden_size, use_batch_norm),
            conv_block(hidden_size, hidden_size, use_batch_norm),
            conv_block(hidden_size, hidden_size, use_batch_norm),
            conv_block(hidden_size, output_size, use_batch_norm),
        )

        self.linear = torch.nn.Linear(output_size, num_classes)

    def forward(self, x):
        out = self.encoder(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def main_worker(gpu, num_gpus_per_node, args):
    global best_loss
    global log_dir
    global writer
    global time_stamp
    global version
    global suffix
    args.gpu = gpu

    suffix = 'Audio' if args.audio else 'MNIST'

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * num_gpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.audio:
        audio_transform = torchvision.transforms.Compose([
            # Transpose from DxT to TxD
            Lambda(lambda mfcc: mfcc.transpose(1, 0)),
            # Add channel dimension
            Lambda(lambda mfcc: mfcc.unsqueeze(0))
        ])

        train_set = FSDDataset(root=os.path.join(args.dir, 'FSDDataset'),
                               train=True,
                               num_mfcc=args.num_mfcc,
                               num_mfcc_time_samples=args.num_mfcc_time_samples,
                               download=True,
                               transform=audio_transform)

        test_set = FSDDataset(root=os.path.join(args.dir, 'FSDDataset'),
                              train=False,
                              num_mfcc=args.num_mfcc,
                              num_mfcc_time_samples=args.num_mfcc_time_samples,
                              download=True,
                              transform=audio_transform)
    else:
        image_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        train_set = torchvision.datasets.MNIST(os.path.join(args.dir, 'MNISTDataset'), train=True,
                                               download=True, transform=image_transform)

        test_set = torchvision.datasets.MNIST(os.path.join(args.dir, 'MNISTDataset'), train=False,
                                              download=True, transform=image_transform)

    # Split to train and validation set. There could be some overlap since we sample at random; use with caution
    train_set, val_set = torch.utils.data.random_split(train_set,
                                                       [ceil(0.9 * len(train_set)), floor(0.1 * len(train_set))])

    print("=> creating digit classification model")
    model = DigitClassifier(use_batch_norm=args.use_batch_norm)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size // num_gpus_per_node)
            args.workers = int((args.workers + num_gpus_per_node - 1) / num_gpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # Define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = utils.checkpoint.load_checkpoint(args.resume, 'cpu')
            else:
                checkpoint = utils.checkpoint.load_checkpoint(args.resume, 'cuda:{}'.format(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            log_dir = checkpoint['log_dir']
            time_stamp = checkpoint['time_stamp']

            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=args.pin_data_to_memory,
        prefetch_factor=args.prefetch_factor, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=args.pin_data_to_memory, prefetch_factor=args.prefetch_factor)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=args.pin_data_to_memory, prefetch_factor=args.prefetch_factor)

    if args.evaluate:
        validate(test_loader, model, criterion, args, prefix='Test: ')
        return

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and
                                                args.rank % num_gpus_per_node == 0):
        if log_dir and args.logging:
            # Use the directory that is stored in checkpoint if we resume training
            writer = SummaryWriter(log_dir=log_dir)
        elif args.logging:
            log_dir = os.path.join('results', 'runs', time_stamp +
                                   '_' + socket.gethostname() +
                                   f'_version-{version}-{suffix}_classification_task')
            writer = SummaryWriter(log_dir=log_dir)

            def pretty_json(hp):
                json_hp = json.dumps(hp, indent=2, sort_keys=False)
                return "".join('\t' + line for line in json_hp.splitlines(True))

            writer.add_text('Info/params', pretty_json(vars(args)))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        current_lr = adjust_learning_rate(optimizer, epoch, args)

        # Train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, args)

        # Evaluate on validation set
        val_loss = validate(val_loader, model, criterion, args)

        # Remember `best_loss` and save checkpoint
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and
                                                    args.rank % num_gpus_per_node == 0):
            if args.logging:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Misc/lr', current_lr, epoch)
                if epoch + 1 == args.epochs:
                    writer.flush()

            utils.checkpoint.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                'log_dir': writer.get_logdir() if args.logging else '',
                'time_stamp': time_stamp,
                'params': args
            }, is_best, filename=os.path.join(
                'results', 'checkpoints', time_stamp + '_' + socket.gethostname() +
                f'_version-{version}-{suffix}_classification_task'))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = utils.meters.AverageMeter('Time', ':6.3f')
    data_time = utils.meters.AverageMeter('Data', ':6.3f')
    losses = utils.meters.AverageMeter('Loss', ':.4e')
    top1 = utils.meters.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.meters.AverageMeter('Acc@5', ':6.2f')
    progress = utils.meters.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # Switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = sample[0], sample[1]

        if torch.cuda.is_available():
            inputs = inputs.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

        # Compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc1, acc5 = utils.metrics.accuracy(torch.nn.Softmax()(outputs), targets, top_k=(1, 5))

        # Record loss
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        # Compute gradient
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Do SGD step
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg


def validate(data_loader, model, criterion, args, prefix="Val: "):
    batch_time = utils.meters.AverageMeter('Time', ':6.3f')
    losses = utils.meters.AverageMeter('Loss', ':.4e')
    top1 = utils.meters.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.meters.AverageMeter('Acc@5', ':6.2f')
    progress = utils.meters.ProgressMeter(
        len(data_loader),
        [batch_time, losses],
        prefix=prefix)

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, sample in enumerate(data_loader):

            inputs, targets = sample[0], sample[1]

            if torch.cuda.is_available():
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)

            # Compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc1, acc5 = utils.metrics.accuracy(torch.nn.Softmax()(outputs), targets, top_k=(1, 5))

            # Record loss
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Loss {losses.avg:.3f} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(losses=losses, top1=top1,
                                                                                          top5=top5))
    return losses.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by X% every N epochs"""
    lr = args.learning_rate * (args.learning_rate_decay ** (epoch // args.decay_learning_rate_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


if __name__ == '__main__':
    main()
