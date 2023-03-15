"""Plot layer outputs of the model for the Omniglot one-shot task with ReLU activations"""

import argparse
import random
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

import utils.checkpoint
from data.omniglot_dataset import OmniglotDataset
from functions.plasticity_functions import InvertedOjaWithSoftUpperBound
from models.network_models import OmniglotOneShotReLU
from models.protonet_models import ProtoNet
from utils.episodic_batch_sampler import EpisodicBatchSampler


def main():
    parser = argparse.ArgumentParser(description='Omniglot one-shot task plotting')
    parser.add_argument('--checkpoint_path', default='', type=str, metavar='PATH',
                        help='Path to checkpoint (default: none)')
    parser.add_argument('--check_params', default=1, type=int, choices=[0, 1], metavar='CHECK_PARAMS',
                        help='When loading from a checkpoint check if the model was trained with the same parameters '
                             'as requested now (default: 1)')

    parser.add_argument('--num_classes', default=5, type=int, metavar='N',
                        help='Number of random classes per sample (default: 5)')

    parser.add_argument('--memory_size', default=100, type=int, metavar='N',
                        help='Size of the memory matrix (default: 100)')
    parser.add_argument('--w_max', default=1.0, type=float, metavar='N',
                        help='Soft maximum of Hebbian weights (default: 1.0)')
    parser.add_argument('--gamma_pos', default=0.1, type=float, metavar='N',
                        help='Write factor of Hebbian rule (default: 0.1)')
    parser.add_argument('--gamma_neg', default=0.1, type=float, metavar='N',
                        help='Forget factor of Hebbian rule (default: 0.1)')

    parser.add_argument('--seed', default=None, type=int, metavar='N',
                        help='Seed for initializing (default: none)')
    parser.add_argument('--sampler_seed', default=42, type=int, metavar='N',
                        help='Seed for episodic batch sampler (default: 42)')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading code
    test_set = OmniglotDataset(mode='test', root='./data')
    test_sampler = EpisodicBatchSampler(test_set.y, num_classes=args.num_classes, batch_size=1, iterations=1,
                                        seed=args.sampler_seed)
    test_loader = torch.utils.data.DataLoader(test_set, batch_sampler=test_sampler)

    # Create SpikingProtoNet
    image_embedding_layer = ProtoNet()

    # Create the model
    model = OmniglotOneShotReLU(
        num_embeddings=args.num_classes,
        output_size=args.num_classes,
        memory_size=args.memory_size,
        image_embedding_layer=image_embedding_layer,
        plasticity_rule=InvertedOjaWithSoftUpperBound(w_max=args.w_max,
                                                      gamma_pos=args.gamma_pos,
                                                      gamma_neg=args.gamma_neg))

    # Load checkpoint
    if args.checkpoint_path:
        print("=> loading checkpoint '{}'".format(args.checkpoint_path))
        checkpoint = utils.checkpoint.load_checkpoint(args.checkpoint_path, device)
        best_acc = checkpoint['best_acc']
        print("Best accuracy {}".format(best_acc))
        if args.check_params:
            for key, val in vars(args).items():
                if key not in ['check_params', 'seed', 'sampler_seed', 'checkpoint_path']:
                    if vars(checkpoint['params'])[key] != val:
                        print("=> You tried to load a model that was trained on different parameters as you requested "
                              "now. You may disable this check by setting `check_params` to 0. Aborting...")
                        sys.exit()

        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('module.'):
                k = k[len('module.'):]  # remove `module.`
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    # Switch to evaluate mode
    model.eval()

    # Get dataset example and run the model
    x, y = next(iter(test_loader))
    x = x.view(1, -1, *x.size()[1:])

    facts = x[:, :args.num_classes]
    query = x[:, -1]

    labels = torch.arange(0, args.num_classes)
    labels = labels.expand(1, args.num_classes).long()

    outputs, encoding_outputs, writing_outputs, reading_outputs = model(facts, labels, query)

    # Get the outputs
    facts_encoded = encoding_outputs[0].detach().numpy()
    labels_encoded = encoding_outputs[1].detach().numpy()
    query_encoded = encoding_outputs[2].detach().numpy()
    mem = writing_outputs[0].detach().numpy()
    write_key = writing_outputs[1].detach().numpy()
    write_val = writing_outputs[2].detach().numpy()
    read_key = reading_outputs[0].detach().numpy()
    read_val = reading_outputs[1].detach().numpy()
    outputs = outputs.detach().numpy()

    # Make some plots
    fig, ax = plt.subplots(nrows=2, ncols=args.num_classes, sharex='all')
    for i in range(args.num_classes):
        img = facts[0][i].numpy()
        ax[0, i].imshow(np.transpose(img, (1, 2, 0)), aspect='equal', cmap='gray', vmin=0, vmax=1)
        ax[0, i].set_axis_off()
        ax[1, i].set_axis_off()

    img = query[0].numpy()
    ax[1, -1].imshow(np.transpose(img, (1, 2, 0)), aspect='equal', cmap='gray', vmin=0, vmax=1)
    plt.tight_layout()

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex='col')
    ax[0].pcolormesh(facts_encoded[0].T, cmap='binary')
    ax[0].set_ylabel('facts')
    ax[1].pcolormesh(labels_encoded[0].T, cmap='binary')
    ax[1].set_ylabel('labels')
    ax[2].pcolormesh(query_encoded[0].T, cmap='binary')
    ax[2].set_ylabel('query')
    plt.tight_layout()

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex='col')
    ax[0].pcolormesh(write_key[0].T, cmap='binary')
    ax[0].set_ylabel('write keys')
    ax[1].pcolormesh(write_val[0].T, cmap='binary')
    ax[1].set_ylabel('write values')
    plt.tight_layout()

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='all')
    ax.matshow(mem[0], cmap='RdBu')
    plt.tight_layout()

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex='col')
    ax[0].pcolormesh(read_key[0].T, cmap='binary')
    ax[0].set_ylabel('read keys')
    ax[1].pcolormesh(read_val[0].T, cmap='binary')
    ax[1].set_ylabel('read values')
    plt.tight_layout()

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='all')
    ax.pcolormesh(outputs[:, None], cmap='binary')
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
