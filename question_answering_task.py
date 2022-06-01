import argparse
import os
import random
import socket
import sys
import time
import warnings
from datetime import datetime
from math import ceil, floor

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

import utils.checkpoint
import utils.meters
import utils.metrics
from data.babi_dataset import BABIDataset
from functions.autograd_functions import SpikeFunction
from functions.plasticity_functions import InvertedOjaWithSoftUpperBound
from models.network_models import QuestionAnswering
from models.neuron_models import IafPscDelta

parser = argparse.ArgumentParser(description='Question answering task training')
parser.add_argument('--task', default=1, choices=range(1, 21), type=int, metavar='TASK',
                    help='The bAbI task (default: 1)')
parser.add_argument('--ten_k', default=1, choices=[0, 1], type=int, metavar='TEN_K',
                    help='Use 10k examples (default: 1)')
parser.add_argument('--add_time_words', default=0, choices=[0, 1], type=int, metavar='ADD_TIME_WORDS',
                    help='Add time word to sentences (default: 0)')
parser.add_argument('--mask_time_words', default=1, choices=[0, 1], type=int, metavar='MASK_TIME_WORDS',
                    help='Prevent learning of time-word encoding (make encoding of time words identity). '
                         'Is set to zero if add_time_words=0 (default: 1)')
parser.add_argument('--learn_encoding', default=1, choices=[0, 1], type=int, metavar='LEARN_ENCODING',
                    help='Learn the sentence encoding or use BoWs. (default: 1)')
parser.add_argument('--sentence_duration', default=100, type=int, metavar='N',
                    help='Number of time steps for each sentence (default: 100)')
parser.add_argument('--max_num_sentences', default=50, type=int, metavar='N',
                    help='Extract only stories with no more than max_num_sentences. '
                         'If None extract all sentences of the stories (default: 50)')
parser.add_argument('--padding', default='pre', choices=['pre', 'post'], type=str, metavar='PADDING',
                    help='Where to pad (default: pre)')

parser.add_argument('--dir', default='./data', type=str, metavar='DIR',
                    help='Path to dataset (default: ./data)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='Number of data loading workers (default: 4)')
parser.add_argument('--prefetch_factor', default=2, type=int, metavar='N',
                    help='Prefetch prefetch_factor * workers examples (default: 2)')
parser.add_argument('--pin_data_to_memory', default=1, choices=[0, 1], type=int, metavar='PIN_DATA_TO_MEMORY',
                    help='Pin data to memory (default: 1)')

parser.add_argument('--embedding_size', default=80, type=int, metavar='N',
                    help='Embedding size (default: 80)')
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
parser.add_argument('--readout_delay', default=30, type=int, metavar='N',
                    help='Synaptic delay of the feedback-connections from value-neurons to key-neurons in the '
                         'reading layer (default: 30)')

parser.add_argument('--thr', default=0.1, type=float, metavar='N',
                    help='Spike threshold (default: 0.1)')
parser.add_argument('--perfect_reset', action='store_true',
                    help='Set the membrane potential to zero after a spike')
parser.add_argument('--refractory_time_steps', default=3, type=int, metavar='N',
                    help='The number of time steps the neuron is refractory (default: 3)')
parser.add_argument('--tau_mem', default=20.0, type=float, metavar='N',
                    help='Neuron membrane time constant (default: 20.0)')
parser.add_argument('--dampening_factor', default=1.0, type=float, metavar='N',
                    help='Scale factor for spike pseudo-derivative (default: 1.0)')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='Number of total epochs to run (default: 200)')
parser.add_argument('--batch_size', default=256, type=int, metavar='N',
                    help='Mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--learning_rate', default=0.003, type=float, metavar='N',
                    help='Initial learning rate (default: 0.003)')
parser.add_argument('--learning_rate_decay', default=0.85, type=float, metavar='N',
                    help='Learning rate decay (default: 0.85)')
parser.add_argument('--decay_learning_rate_every', default=20, type=int, metavar='N',
                    help='Decay the learning rate every N epochs (default: 20)')
parser.add_argument('--max_grad_norm', default=40.0, type=float, metavar='N',
                    help='Gradients with an L2 norm larger than max_grad_norm will be clipped '
                         'to have an L2 norm of max_grad_norm. If None, then the gradient will '
                         'not be clipped. (default: 40.0)')
parser.add_argument('--l2', default=1e-5, type=float, metavar='N',
                    help='L2 rate regularization factor (default: 1e-5)')
parser.add_argument('--target_rate', default=0.0, type=float, metavar='N',
                    help='Target firing rate in Hz for L2 regularization (default: 0.0)')

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
parser.add_argument('--gpu', default=None, type=int, metavar='N',
                    help='GPU id to use (default: none)')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc = 0
log_dir = ''
writer = None
time_stamp = datetime.now().strftime('%b%d_%H-%M-%S')
suffix = ''

with open('version.txt') as f:
    version = f.readline()


def main():
    args = parser.parse_args()

    args.ten_k = True if args.ten_k else False
    args.add_time_words = True if args.add_time_words else False
    args.mask_time_words = True if args.mask_time_words else False
    args.learn_encoding = True if args.learn_encoding else False
    args.pin_data_to_memory = True if args.pin_data_to_memory else False

    if not args.add_time_words:
        args.mask_time_words = False

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
    global best_acc
    global log_dir
    global writer
    global time_stamp
    global version
    global suffix
    args.gpu = gpu

    if args.learn_encoding:
        if args.add_time_words:
            suffix = 'le-te'
        else:
            suffix = 'le'
    else:
        if args.add_time_words:
            suffix = 'bow-te'
        else:
            suffix = 'bow'

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
    train_set = BABIDataset(root=args.dir, task=args.task, train=True, ten_k=args.ten_k,
                            max_num_sentences=args.max_num_sentences, download=True)

    test_set = BABIDataset(root=args.dir, task=args.task, train=False, ten_k=args.ten_k,
                           max_num_sentences=args.max_num_sentences, download=False)

    stories = train_set.stories + test_set.stories
    max_num_words = max(train_set.stats['max_num_words'], test_set.stats['max_num_words'])
    max_num_sentences = max(train_set.stats['max_num_sentences'], test_set.stats['max_num_sentences'])
    vocab, vocab_size = BABIDataset.build_vocab(stories, max_num_sentences, add_time_words=args.add_time_words)

    num_embeddings = len(vocab)  # This is the length of the vocab with time-words
    sentence_size = max_num_words + 1 if args.add_time_words else max_num_words
    train_set.vectorize_stories(vocab, sentence_size, add_time_words=args.add_time_words, padding=args.padding)
    test_set.vectorize_stories(vocab, sentence_size, add_time_words=args.add_time_words, padding=args.padding)

    # Split to train and validation set
    train_set, val_set = torch.utils.data.random_split(train_set, [ceil(0.9*len(train_set)), floor(0.1*len(train_set))])

    # Create model
    print("=> creating model '{model_name}'".format(model_name=QuestionAnswering.__name__))
    model = QuestionAnswering(
        input_size=sentence_size,
        output_size=vocab_size,
        num_embeddings=num_embeddings,
        embedding_size=args.embedding_size,
        memory_size=args.memory_size,
        mask_time_words=args.mask_time_words,
        learn_encoding=args.learn_encoding,
        num_time_steps=args.sentence_duration,
        readout_delay=args.readout_delay,
        tau_trace=args.tau_trace,
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
            args.batch_size = int(args.batch_size / num_gpus_per_node)
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
                checkpoint = utils.checkpoint.load_checkpoint(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint = utils.checkpoint.load_checkpoint(args.resume, 'cuda:{}'.format(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            log_dir = checkpoint['log_dir']
            time_stamp = checkpoint['time_stamp']

            # Checkpoint parameters have to match current parameters. If not, abort.
            ignore_keys = ['workers', 'prefetch_factor', 'pin_data_to_memory', 'epochs', 'start_epoch', 'resume',
                           'evaluate', 'logging', 'print_freq', 'world_size', 'rank', 'dist_url', 'dist_backend',
                           'seed', 'gpu', 'multiprocessing_distributed']
            if args.evaluate:
                ignore_keys.append('batch_size')

            for key, val in vars(checkpoint['params']).items():
                if key not in ignore_keys:
                    if vars(args)[key] != val:
                        print("=> You tried to restart training of a model that was trained with different parameters "
                              "as you requested now. Aborting...")
                        sys.exit()

            if args.gpu is not None:
                # best_acc may be from a checkpoint from a different GPU
                best_acc = best_acc.to(args.gpu)
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
                                   '_' + socket.gethostname() + f'_version-{version}-{suffix}_babi_task{args.task}')
            writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        current_lr = adjust_learning_rate(optimizer, epoch, args)

        # Train for one epoch
        train_loss, reg_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args)

        # Evaluate on validation set
        val_loss, val_acc = validate(val_loader, model, criterion, args)

        # Remember best acc@1 and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and
                                                    args.rank % num_gpus_per_node == 0):
            if args.logging:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Acc/train', train_acc, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Acc/val', val_acc, epoch)
                writer.add_scalar('Misc/lr', current_lr, epoch)
                writer.add_scalar('Misc/reg_loss', reg_loss, epoch)
                if epoch + 1 == args.epochs:
                    writer.flush()

            utils.checkpoint.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'log_dir': writer.get_logdir() if args.logging else '',
                'time_stamp': time_stamp,
                'params': args
            }, is_best, filename=os.path.join(
                'results', 'checkpoints', time_stamp +
                '_' + socket.gethostname() + f'_version-{version}-{suffix}_babi_task{args.task}'))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = utils.meters.AverageMeter('Time', ':6.3f')
    data_time = utils.meters.AverageMeter('Data', ':6.3f')
    losses = utils.meters.AverageMeter('Loss', ':.4e')
    reg_losses = utils.meters.AverageMeter('RegLoss', ':.4e')
    top1 = utils.meters.AverageMeter('Acc', ':6.2f')
    progress = utils.meters.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # Switch to train mode
    model.train()

    end = time.time()
    for i, (sample, story_length) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        story, query, answer = sample['story'], sample['query'], sample['answer']

        if args.gpu is not None:
            story = story.cuda(args.gpu, non_blocking=True)
            query = query.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            answer = answer.cuda(args.gpu, non_blocking=True)
            story_length = story_length.cuda(args.gpu, non_blocking=True)

        # Compute output
        output, encoding_outputs, writing_outputs, reading_outputs = model(story, query)
        loss = criterion(output, answer)

        # Regularization
        def compute_l2_loss(x, target, weight):
            if isinstance(weight, torch.Tensor):
                mean = torch.mean(torch.sum(x, dim=1) / weight.unsqueeze(1), dim=0)
            else:
                mean = torch.mean(torch.sum(x, dim=1) / weight, dim=0)

            return torch.mean((mean - target)**2)

        l2_act_reg_loss = 0
        weight_query = 1e-3 * args.sentence_duration
        weight_story = 1e-3 * story_length * args.sentence_duration
        l2_act_reg_loss += compute_l2_loss(encoding_outputs[0], args.target_rate, weight=weight_story)
        l2_act_reg_loss += compute_l2_loss(encoding_outputs[1], args.target_rate, weight=weight_query)

        l2_act_reg_loss += compute_l2_loss(writing_outputs[1], args.target_rate, weight=weight_story)
        l2_act_reg_loss += compute_l2_loss(writing_outputs[2], args.target_rate, weight=weight_story)

        l2_act_reg_loss += compute_l2_loss(reading_outputs[0], args.target_rate, weight=weight_query)
        l2_act_reg_loss += compute_l2_loss(reading_outputs[1], args.target_rate, weight=weight_query)

        act_reg_loss = args.l2 * l2_act_reg_loss

        if epoch > 0:
            # Start regularizing after an initial training period
            loss += act_reg_loss

        # Measure accuracy and record loss
        acc = utils.metrics.accuracy(output, answer, top_k=(1,))

        losses.update(loss.item(), story.size(0))
        reg_losses.update(act_reg_loss, story.size(0))
        top1.update(acc[0], story.size(0))

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

    return losses.avg, reg_losses.avg, top1.avg


def validate(data_loader, model, criterion, args, prefix="Val: "):
    batch_time = utils.meters.AverageMeter('Time', ':6.3f')
    losses = utils.meters.AverageMeter('Loss', ':.4e')
    top1 = utils.meters.AverageMeter('Acc', ':6.2f')
    progress = utils.meters.ProgressMeter(
        len(data_loader),
        [batch_time, losses, top1],
        prefix=prefix)

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (sample, story_length) in enumerate(data_loader):
            story, query, answer = sample['story'], sample['query'], sample['answer']

            if args.gpu is not None:
                story = story.cuda(args.gpu, non_blocking=True)
                query = query.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                answer = answer.cuda(args.gpu, non_blocking=True)

            # Compute output
            output, *_ = model(story, query)
            loss = criterion(output, answer)

            # Measure accuracy and record loss
            acc = utils.metrics.accuracy(output, answer, top_k=(1,))
            losses.update(loss.item(), story.size(0))
            top1.update(acc[0], story.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc {top1.avg:.3f}'.format(top1=top1))

    return losses.avg, top1.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by X% every N epochs"""
    lr = args.learning_rate * (args.learning_rate_decay ** (epoch // args.decay_learning_rate_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


if __name__ == '__main__':
    main()
