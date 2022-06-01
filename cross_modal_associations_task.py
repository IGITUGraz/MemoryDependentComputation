import argparse
import json
import os
import random
import socket
import sys
import time
import warnings
from datetime import datetime
from math import ceil, floor

import numpy as np
import torch
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
from data.cross_modal_associations_dataset import CrossModalAssociationsDataset
from functions.autograd_functions import SpikeFunction
from functions.plasticity_functions import InvertedOjaWithSoftUpperBound
from models.network_models import CrossModalAssociations
from models.neuron_models import IafPscDelta
from models.spiking_protonet import SpikingProtoNet

parser = argparse.ArgumentParser(description='Cross-modal associations task training')
parser.add_argument('--sequence_length', default=3, type=int, metavar='N',
                    help='Number of audio-image pairs per example (default: 3)')
parser.add_argument('--train_classes', default=list(range(10)), nargs='+', type=int, metavar='TRAIN_CLASSES',
                    help='Classes used during training (default: list(range(10))')
parser.add_argument('--test_classes', default=list(range(10)), nargs='+', type=int, metavar='TEST_CLASSES',
                    help='Classes used during inference (default: list(range(10))')
parser.add_argument('--dataset_size', default=10000, type=int, metavar='DATASET_SIZE',
                    help='Number of examples in the dataset (default: 10000)')
parser.add_argument('--num_mfcc', default=20, type=int, metavar='NUM_MFCC',
                    help='Number of Mel-frequency cepstrum coefficients (default: 20)')
parser.add_argument('--num_mfcc_time_samples', default=30, type=int, metavar='NUM_MFCC_TIME_SAMPLES',
                    help='Number of time samples for the Mel-frequency cepstrum coefficients (default: 30)')
parser.add_argument('--num_time_steps', default=100, type=int, metavar='N',
                    help='Number of time steps for each item in the sequence (default: 100)')

parser.add_argument('--dir', default='./data', type=str, metavar='DIR',
                    help='Path to dataset (default: ./data)')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='Number of data loading workers; set to 0 due to pickling issue (default: 4)')
parser.add_argument('--prefetch_factor', default=2, type=int, metavar='N',
                    help='Prefetch prefetch_factor * workers examples (default: 2)')
parser.add_argument('--pin_data_to_memory', default=1, choices=[0, 1], type=int, metavar='PIN_DATA_TO_MEMORY',
                    help='Pin data to memory (default: 1)')

parser.add_argument('--embedding_size', default=64, type=int, metavar='N',
                    help='Embedding size (default: 64)')
parser.add_argument('--memory_size', default=100, type=int, metavar='N',
                    help='Size of the memory matrix (default: 100)')
parser.add_argument('--w_max', default=1.0, type=float, metavar='N',
                    help='Soft maximum of Hebbian weights (default: 1.0)')
parser.add_argument('--gamma_pos', default=0.3, type=float, metavar='N',
                    help='Write factor of Hebbian rule (default: 0.3)')
parser.add_argument('--gamma_neg', default=0.3, type=float, metavar='N',
                    help='Forget factor of Hebbian rule (default: 0.3)')
parser.add_argument('--tau_trace', default=20.0, type=float, metavar='N',
                    help='Time constant of key- and value-trace (default: 20.0)')
parser.add_argument('--readout_delay', default=1, type=int, metavar='N',
                    help='Synaptic delay of the feedback-connections from value-neurons to key-neurons in the '
                         'reading layer (default: 1)')

parser.add_argument('--thr', default=0.05, type=float, metavar='N',
                    help='Spike threshold (default: 0.05)')
parser.add_argument('--perfect_reset', action='store_true',
                    help='Set the membrane potential to zero after a spike')
parser.add_argument('--refractory_time_steps', default=3, type=int, metavar='N',
                    help='The number of time steps the neuron is refractory (default: 3)')
parser.add_argument('--tau_mem', default=20.0, type=float, metavar='N',
                    help='Neuron membrane time constant (default: 20.0)')
parser.add_argument('--dampening_factor', default=1.0, type=float, metavar='N',
                    help='Scale factor for spike pseudo-derivative (default: 1.0)')

parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='Number of total epochs to run (default: 120)')
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
parser.add_argument('--max_grad_norm', default=40.0, type=float, metavar='N',
                    help='Gradients with an L2 norm larger than max_grad_norm will be clipped '
                         'to have an L2 norm of max_grad_norm. If None, then the gradient will '
                         'not be clipped. (default: 40.0)')
parser.add_argument('--l2', default=1e-7, type=float, metavar='N',
                    help='L2 rate regularization factor (default: 1e-7)')
parser.add_argument('--target_rate', default=0.0, type=float, metavar='N',
                    help='Target firing rate in Hz for L2 regularization (default: 0.0)')

parser.add_argument('--use_pretrained_protonet', default=1, choices=[0, 1], type=int, metavar='USE_PRETRAINED_PROTONET',
                    help='Use Conv-nets that were pretrained somehow (default: 1)')
parser.add_argument('--image_protonet_path', default='results/checkpoints/cross-modal-associations-task-image'
                                                     '-protonet-checkpoint.tar',
                    type=str, metavar='PATH', help='Path to the MNIST ProtoNet checkpoint (default: none)')
parser.add_argument('--audio_protonet_path', default='results/checkpoints/cross-modal-associations-task-audio'
                                                     '-protonet-checkpoint.tar',
                    type=str, metavar='PATH', help='Path to the Audio ProtoNet checkpoint (default: none)')
parser.add_argument('--freeze_protonet', default=0, choices=[0, 1], type=int,
                    help='Freeze pre-trained ProtoNets after conversion (default: 0)')
parser.add_argument('--learn_if_thresholds', action='store_true',
                    help='Learn IF neuron thresholds of the spiking CNN via threshold balancing (if never done yet)')
parser.add_argument('--fix_cnn_thresholds', action='store_false', help='Do not adjust firing threshold after conversion'
                                                                       '(default: will adjust v_th via a bias)')

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='Manual epoch number (useful on restarts, default: 0)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', action='store_true',
                    help='Evaluate the model on the test set')

parser.add_argument('--logging', action='store_true',
                    help='Write tensorboard logs')
parser.add_argument('--print_freq', default=1, type=int, metavar='N',
                    help='Print frequency (default: 1)')
parser.add_argument('--world_size', default=-1, type=int, metavar='N',
                    help='Number of nodes for distributed training (default: -1)')
parser.add_argument('--rank', default=-1, type=int, metavar='N',
                    help='Node rank for distributed training (default: -1)')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:23456', type=str, metavar='DIST_URL',
                    help='URL used to set up distributed training (default: tcp://127.0.0.1:23456)')
parser.add_argument('--dist_backend', default='nccl', choices=['nccl', 'mpi', 'gloo'], type=str, metavar='DIST_BACKEND',
                    help='Distributed backend to use (default: nccl)')
parser.add_argument('--seed', default=None, type=int, metavar='N',
                    help='Seed for initializing training (default: none)')
parser.add_argument('--data_seed', default=None, type=int, metavar='N',
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


def main_worker(gpu, num_gpus_per_node, args):
    global best_loss
    global log_dir
    global writer
    global time_stamp
    global version
    global suffix
    args.gpu = gpu

    suffix = f'num_pairs{args.sequence_length}'

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

    # Data loading code
    audio_transform = torchvision.transforms.Compose([
        # Transpose from DxT to TxD
        Lambda(lambda mfcc: mfcc.transpose(1, 0)),
        # Add channel dimension
        Lambda(lambda mfcc: mfcc.unsqueeze(0))
    ])

    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    train_set = CrossModalAssociationsDataset(root=args.dir, dataset_size=args.dataset_size,
                                              sequence_length=args.sequence_length, num_mfcc=args.num_mfcc,
                                              num_mfcc_time_samples=args.num_mfcc_time_samples, train=True,
                                              classes=args.train_classes, image_transform=image_transform,
                                              audio_transform=audio_transform,
                                              rng=np.random.default_rng(args.data_seed))

    test_set = CrossModalAssociationsDataset(root=args.dir, dataset_size=args.dataset_size,
                                             sequence_length=args.sequence_length,
                                             num_mfcc=args.num_mfcc, num_mfcc_time_samples=args.num_mfcc_time_samples,
                                             train=False, classes=args.test_classes, image_transform=image_transform,
                                             audio_transform=audio_transform, rng=np.random.default_rng(args.data_seed))

    # Split to train and validation set. There could be some overlap since we sample at random; use with caution
    train_set, val_set = torch.utils.data.random_split(train_set, [ceil(0.9*len(train_set)), floor(0.1*len(train_set))])

    if args.use_pretrained_protonet:
        image_path = args.image_protonet_path
        audio_path = args.audio_protonet_path
        print("=> loading Image ProtoNet checkpoint '{}'".format(image_path))
        print("=> loading Audio ProtoNet checkpoint '{}'".format(audio_path))
        if args.gpu is None:
            image_protonet_checkpoint = utils.checkpoint.load_checkpoint(image_path, 'cpu')
            audio_protonet_checkpoint = utils.checkpoint.load_checkpoint(audio_path, 'cpu')
        else:
            image_protonet_checkpoint = utils.checkpoint.load_checkpoint(image_path, 'cuda:{}'.format(args.gpu))
            audio_protonet_checkpoint = utils.checkpoint.load_checkpoint(audio_path, 'cuda:{}'.format(args.gpu))

        image_embedding_layer = SpikingProtoNet(IafPscDelta(thr=args.thr,
                                                            perfect_reset=args.perfect_reset,
                                                            refractory_time_steps=args.refractory_time_steps,
                                                            tau_mem=args.tau_mem,
                                                            spike_function=SpikeFunction,
                                                            dampening_factor=args.dampening_factor),
                                                weight_dict=image_protonet_checkpoint['state_dict'],
                                                input_size=784,
                                                output_size=args.embedding_size,
                                                num_time_steps=args.num_time_steps,
                                                refractory_time_steps=args.refractory_time_steps,
                                                use_bias=args.fix_cnn_thresholds)

        mfcc_embedding_layer = SpikingProtoNet(IafPscDelta(thr=args.thr,
                                                           perfect_reset=args.perfect_reset,
                                                           refractory_time_steps=args.refractory_time_steps,
                                                           tau_mem=args.tau_mem,
                                                           spike_function=SpikeFunction,
                                                           dampening_factor=args.dampening_factor),
                                               weight_dict=audio_protonet_checkpoint['state_dict'],
                                               input_size=600,
                                               output_size=args.embedding_size,
                                               num_time_steps=args.num_time_steps,
                                               refractory_time_steps=args.refractory_time_steps,
                                               use_bias=args.fix_cnn_thresholds)

        if args.learn_if_thresholds:
            print('This may take time as it will go through the whole training set.')
            aux_train_loader = torch.utils.data.DataLoader(train_set, batch_size=64)
            with torch.no_grad():
                for i, aux_sample in enumerate(aux_train_loader):
                    mfcc_seq, image_seq, _, _, _ = aux_sample
                    mfcc_seq = mfcc_seq.cuda(args.gpu, non_blocking=True) if args.gpu is not None else mfcc_seq
                    image_seq = image_seq.cuda(args.gpu, non_blocking=True) if args.gpu is not None else image_seq
                    for t in range(mfcc_seq.shape[1]):
                        mfcc_embedding_layer.threshold_balancing(None, mfcc_seq.select(1, t))
                        image_embedding_layer.threshold_balancing(None, image_seq.select(1, t))
                        print(str(i), image_embedding_layer.layer_v_th)
                        print(str(i), mfcc_embedding_layer.layer_v_th)
        else:
            image_embedding_layer.threshold_balancing([1.8209, 11.5916, 4.1207, 2.6341])
            mfcc_embedding_layer.threshold_balancing([2.1098, 3.9579, 3.1872, 1.6841])

        if args.freeze_protonet:
            for param in image_embedding_layer.parameters():
                param.requires_grad = False
            for param in mfcc_embedding_layer.parameters():
                param.requires_grad = False
    else:
        # Create ProtoNet
        image_embedding_layer = SpikingProtoNet(IafPscDelta(thr=args.thr,
                                                            perfect_reset=args.perfect_reset,
                                                            refractory_time_steps=args.refractory_time_steps,
                                                            tau_mem=args.tau_mem,
                                                            spike_function=SpikeFunction,
                                                            dampening_factor=args.dampening_factor),
                                                input_size=784,
                                                output_size=args.embedding_size,
                                                num_time_steps=args.num_time_steps,
                                                refractory_time_steps=args.refractory_time_steps,
                                                use_bias=args.fix_cnn_thresholds)

        mfcc_embedding_layer = SpikingProtoNet(IafPscDelta(thr=args.thr,
                                                           perfect_reset=args.perfect_reset,
                                                           refractory_time_steps=args.refractory_time_steps,
                                                           tau_mem=args.tau_mem,
                                                           spike_function=SpikeFunction,
                                                           dampening_factor=args.dampening_factor),
                                               input_size=600,
                                               output_size=args.embedding_size,
                                               num_time_steps=args.num_time_steps,
                                               refractory_time_steps=args.refractory_time_steps,
                                               use_bias=args.fix_cnn_thresholds)

        image_embedding_layer.threshold_balancing([args.thr, args.thr, args.thr, args.thr])
        mfcc_embedding_layer.threshold_balancing([args.thr, args.thr, args.thr, args.thr])

    # Create model
    print("=> creating model '{model_name}'".format(model_name=CrossModalAssociations.__name__))
    model = CrossModalAssociations(
        output_size=784,
        memory_size=args.memory_size,
        num_time_steps=args.num_time_steps,
        readout_delay=args.readout_delay,
        tau_trace=args.tau_trace,
        image_embedding_layer=image_embedding_layer,
        mfcc_embedding_layer=mfcc_embedding_layer,
        plasticity_rule=InvertedOjaWithSoftUpperBound(w_max=args.w_max,
                                                      gamma_pos=args.gamma_pos,
                                                      gamma_neg=args.gamma_neg),
        dynamics=IafPscDelta(thr=args.thr,
                             perfect_reset=args.perfect_reset,
                             refractory_time_steps=args.refractory_time_steps,
                             tau_mem=args.tau_mem,
                             spike_function=SpikeFunction,
                             dampening_factor=args.dampening_factor))

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
    criterion = torch.nn.MSELoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = utils.checkpoint.load_checkpoint(args.resume, 'cpu')
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint = utils.checkpoint.load_checkpoint(args.resume, 'cuda:{}'.format(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            log_dir = checkpoint['log_dir']
            time_stamp = checkpoint['time_stamp']

            # Checkpoint parameters have to match current parameters. If not, abort.
            ignore_keys = ['workers', 'prefetch_factor', 'pin_data_to_memory', 'epochs', 'start_epoch', 'resume',
                           'evaluate', 'logging', 'print_freq', 'world_size', 'rank', 'dist_url', 'dist_backend',
                           'seed', 'data_seed', 'gpu', 'multiprocessing_distributed', 'distributed', 'dir']
            if args.evaluate:
                ignore_keys.append('batch_size')

            for key, val in vars(checkpoint['params']).items():
                if key not in ignore_keys:
                    if vars(args)[key] != val:
                        print("=> You tried to restart training of a model that was trained with different parameters "
                              "as you requested now. Aborting...")
                        sys.exit()

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
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
                                   f'_version-{version}-{suffix}_cross-modal-associations_task')
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
        train_loss, reg_loss = train(train_loader, model, criterion, optimizer, epoch, args)

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
                writer.add_scalar('Misc/reg_loss', reg_loss, epoch)
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
                'results', 'checkpoints',
                time_stamp + '_' + socket.gethostname() + f'_version-{version}-{suffix}_cross-modal-associations_task'))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = utils.meters.AverageMeter('Time', ':6.3f')
    data_time = utils.meters.AverageMeter('Data', ':6.3f')
    losses = utils.meters.AverageMeter('Loss', ':.4e')
    reg_losses = utils.meters.AverageMeter('RegLoss', ':.4e')
    progress = utils.meters.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # Switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        mfcc_sequence, image_sequence, mfcc_query, image_target, targets = sample

        if args.gpu is not None:
            mfcc_sequence = mfcc_sequence.cuda(args.gpu, non_blocking=True)
            image_sequence = image_sequence.cuda(args.gpu, non_blocking=True)
            mfcc_query = mfcc_query.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            image_target = image_target.cuda(args.gpu, non_blocking=True)

        # Compute output
        output, encoding_outputs, writing_outputs, reading_outputs, decoder_outputs = model(
            mfcc_sequence, image_sequence, mfcc_query)
        output = output.view(-1, 1, 28, 28)
        loss = criterion(output, image_target)

        # Regularization
        def compute_l2_loss(x, target, weight):
            if isinstance(weight, torch.Tensor):
                mean = torch.mean(torch.sum(x, dim=1) / weight.unsqueeze(1), dim=0)
            else:
                mean = torch.mean(torch.sum(x, dim=1) / weight, dim=0)

            return torch.mean((mean - target) ** 2)

        l2_act_reg_loss = 0
        weight_query = 1e-3 * args.num_time_steps
        weight_facts = 1e-3 * args.sequence_length * args.num_time_steps
        l2_act_reg_loss += compute_l2_loss(encoding_outputs[0], args.target_rate, weight=weight_facts)
        l2_act_reg_loss += compute_l2_loss(encoding_outputs[1], args.target_rate, weight=weight_facts)
        l2_act_reg_loss += compute_l2_loss(encoding_outputs[2], args.target_rate, weight=weight_query)

        l2_act_reg_loss += compute_l2_loss(writing_outputs[1], args.target_rate, weight=weight_facts)
        l2_act_reg_loss += compute_l2_loss(writing_outputs[2], args.target_rate, weight=weight_facts)

        l2_act_reg_loss += compute_l2_loss(reading_outputs[0], args.target_rate, weight=weight_query)
        l2_act_reg_loss += compute_l2_loss(reading_outputs[1], args.target_rate, weight=weight_query)

        l2_act_reg_loss += compute_l2_loss(decoder_outputs[0], args.target_rate, weight=weight_query)
        l2_act_reg_loss += compute_l2_loss(decoder_outputs[1], args.target_rate, weight=weight_query)

        act_reg_loss = args.l2 * l2_act_reg_loss
        loss += act_reg_loss

        # Record loss
        losses.update(loss.item(), mfcc_sequence.size(0))
        reg_losses.update(act_reg_loss, mfcc_sequence.size(0))

        # Compute gradient
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.max_grad_norm is not None:
            # Clip L2 norm of gradient to max_grad_norm
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.max_grad_norm)

        # Do SGD step
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, reg_losses.avg


def validate(data_loader, model, criterion, args, prefix="Val: "):
    batch_time = utils.meters.AverageMeter('Time', ':6.3f')
    losses = utils.meters.AverageMeter('Loss', ':.4e')
    progress = utils.meters.ProgressMeter(
        len(data_loader),
        [batch_time, losses],
        prefix=prefix)

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, sample in enumerate(data_loader):
            mfcc_sequence, image_sequence, mfcc_query, image_target, targets = sample

            if args.gpu is not None:
                mfcc_sequence = mfcc_sequence.cuda(args.gpu, non_blocking=True)
                image_sequence = image_sequence.cuda(args.gpu, non_blocking=True)
                mfcc_query = mfcc_query.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                image_target = image_target.cuda(args.gpu, non_blocking=True)

            # Compute output
            output, *_ = model(mfcc_sequence, image_sequence, mfcc_query)
            output = output.view(-1, 1, 28, 28)
            loss = criterion(output, image_target)

            # Record loss
            losses.update(loss.item(), mfcc_sequence.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Loss {losses.avg:.3f}'.format(losses=losses))

    return losses.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by X% every N epochs"""
    lr = args.learning_rate * (args.learning_rate_decay ** (epoch // args.decay_learning_rate_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


if __name__ == '__main__':
    main()
