"""Plot layer outputs of the model for the cross-modal associations task with ReLU activations"""

import argparse
import random
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional
import torch.backends.cudnn as cudnn
import torchvision
from torchvision.transforms import Lambda

import utils.checkpoint
from data.cross_modal_associations_dataset import CrossModalAssociationsDataset
from functions.plasticity_functions import InvertedOjaWithSoftUpperBound
from models.network_models import CrossModalAssociationsReLU
from models.protonet_models import ProtoNet


def main():
    parser = argparse.ArgumentParser(description='Cross-modal associations task plotting')
    parser.add_argument('--checkpoint_path', default='', type=str, metavar='PATH',
                        help='Path to checkpoint (default: none)')
    parser.add_argument('--check_params', default=1, type=int, choices=[0, 1], metavar='CHECK_PARAMS',
                        help='When loading from a checkpoint check if the model was trained with the same parameters '
                             'as requested now (default: 1)')

    parser.add_argument('--sequence_length', default=3, type=int, metavar='N',
                        help='Number of audio-image pairs per example (default: 3)')
    parser.add_argument('--test_classes', default=list(range(10)), nargs='+', type=int, metavar='TEST_CLASSES',
                        help='Classes used during inference (default: list(range(10))')
    parser.add_argument('--dataset_size', default=10000, type=int, metavar='DATASET_SIZE',
                        help='Number of examples in the dataset (default: 10000)')
    parser.add_argument('--num_mfcc', default=20, type=int, metavar='NUM_MFCC',
                        help='Number of Mel-frequency cepstrum coefficients (default: 20)')
    parser.add_argument('--num_mfcc_time_samples', default=30, type=int, metavar='NUM_MFCC_TIME_SAMPLES',
                        help='Number of time samples for the Mel-frequency cepstrum coefficients (default: 30)')

    parser.add_argument('--embedding_size', default=64, type=int, metavar='N',
                        help='Embedding size (default: 64)')
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
    parser.add_argument('--data_seed', default=None, type=int, metavar='N',
                        help='Seed for the dataset (default: none)')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    test_set = CrossModalAssociationsDataset(root='./data', dataset_size=args.dataset_size,
                                             sequence_length=args.sequence_length, num_mfcc=args.num_mfcc,
                                             num_mfcc_time_samples=args.num_mfcc_time_samples, train=False,
                                             classes=args.test_classes, image_transform=image_transform,
                                             audio_transform=audio_transform, rng=np.random.default_rng(args.data_seed))

    # Create ProtoNetSpiking
    image_embedding_layer = ProtoNet()

    mfcc_embedding_layer = ProtoNet()

    # Create the model
    model = CrossModalAssociationsReLU(
        output_size=784,
        memory_size=args.memory_size,
        mfcc_embedding_layer=mfcc_embedding_layer,
        image_embedding_layer=image_embedding_layer,
        plasticity_rule=InvertedOjaWithSoftUpperBound(w_max=args.w_max,
                                                      gamma_pos=args.gamma_pos,
                                                      gamma_neg=args.gamma_neg))

    # Load checkpoint
    if args.checkpoint_path:
        print("=> loading checkpoint '{}'".format(args.checkpoint_path))
        checkpoint = utils.checkpoint.load_checkpoint(args.checkpoint_path, device)
        best_loss = checkpoint['best_loss']
        epoch = checkpoint['epoch']
        print("Best loss {}".format(best_loss))
        print("Epoch {}".format(epoch))
        if args.check_params:
            for key, val in vars(args).items():
                if key not in ['check_params', 'seed', 'data_seed', 'checkpoint_path']:
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
    mfcc_sequence, image_sequence, mfcc_query, image_target, targets = test_set[0]
    mfcc_sequence = mfcc_sequence.unsqueeze(0)
    image_sequence = image_sequence.unsqueeze(0)
    mfcc_query = mfcc_query.unsqueeze(0)

    outputs, encoding_outputs, writing_outputs, reading_outputs = model(
        mfcc_sequence, image_sequence, mfcc_query)

    # Get the outputs
    mfcc_encoded = encoding_outputs[0].detach().numpy()
    images_encoded = encoding_outputs[1].detach().numpy()
    query_encoded = encoding_outputs[2].detach().numpy()
    mem = writing_outputs[0].detach().numpy()
    write_key = writing_outputs[1].detach().numpy()
    write_val = writing_outputs[2].detach().numpy()
    read_key = reading_outputs[0].detach().numpy()
    read_val = reading_outputs[1].detach().numpy()
    outputs = outputs.view(1, 28, 28).detach().numpy()

    # Make some plots
    fig, ax = plt.subplots(nrows=2, ncols=mfcc_sequence.size()[1] + 1, sharex='all')
    for i in range(mfcc_sequence.size()[1]):
        image = mfcc_sequence[0][i].numpy()
        ax[0, i].imshow(np.transpose(image, (2, 1, 0)), interpolation='nearest', cmap='viridis', origin='lower')
        ax[0, i].set(title='Digit {}'.format(targets[0][i]))

        image = image_sequence[0][i].numpy()
        ax[1, i].imshow(np.transpose(image, (1, 2, 0)), aspect='equal', cmap='gray', vmin=0, vmax=1)
        ax[0, i].set_axis_off()
        ax[1, i].set_axis_off()

    image = mfcc_query[0].numpy()
    ax[0, -1].imshow(np.transpose(image, (2, 1, 0)), interpolation='nearest', cmap='viridis', origin='lower')
    ax[0, -1].set(title='Query digit {}'.format(targets[2]))

    ax[1, -1].imshow(np.transpose(outputs, (1, 2, 0)),
                     aspect='equal', cmap='gray', vmin=np.min(outputs), vmax=np.max(outputs))
    ax[1, -1].set(title='Reconstructed image')

    ax[0, -1].set_axis_off()
    ax[1, -1].set_axis_off()
    plt.tight_layout()

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex='col')
    ax[0].pcolormesh(mfcc_encoded[0].T, cmap='binary')
    ax[0].set_ylabel('mfcc')
    ax[1].pcolormesh(images_encoded[0].T, cmap='binary')
    ax[1].set_ylabel('images')
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
    ax.imshow(np.transpose(outputs, (1, 2, 0)), aspect='equal', cmap='gray', vmin=np.min(outputs), vmax=np.max(outputs))
    ax.set(title='Reconstructed image')
    plt.tight_layout()

    msd_target = []
    msd_other1 = []
    msd_other2 = []
    for i in range(1000):
        mfcc_sequence, image_sequence, mfcc_query, image_target, targets = test_set[0]
        mfcc_sequence = mfcc_sequence.unsqueeze(0)
        image_sequence = image_sequence.unsqueeze(0)
        mfcc_query = mfcc_query.unsqueeze(0)

        # targets: (mfcc_class_sequence, image_class_sequence, mfcc_query_class, image_target_class)
        image_classes = np.array(targets[1])
        target_class = targets[3]

        outputs, *_ = model(mfcc_sequence, image_sequence, mfcc_query)
        outputs = outputs.view(1, 28, 28).detach()

        idc = np.where(image_classes != target_class)[0]
        msd_target.append(torch.nn.functional.mse_loss(outputs, image_target).item())
        msd_other1.append(torch.nn.functional.mse_loss(outputs, image_sequence[:, idc[0]].squeeze(0)).item())
        msd_other2.append(torch.nn.functional.mse_loss(outputs, image_sequence[:, idc[1]].squeeze(0)).item())

    print('Target: ', np.mean(msd_target), '+-', np.std(msd_target))
    print('Target median: ', np.median(msd_target), np.percentile(msd_target, [25, 75]))

    print('Other 1: ', np.mean(msd_other1), '+-', np.std(msd_other1))
    print('Other 1 median: ', np.median(msd_other1), np.percentile(msd_other1, [25, 75]))
    print('Other 2: ', np.mean(msd_other2), '+-', np.std(msd_other2))
    print('Other 2 median: ', np.median(msd_other2), np.percentile(msd_other2, [25, 75]))

    msd_other = np.concatenate((msd_other1, msd_other2))
    print('Other: ', np.mean(msd_other), '+-', np.std(msd_other))
    print('Other median: ', np.median(msd_other), np.percentile(msd_other, [25, 75]))

    plt.figure()
    plt.hist(msd_target, density=True, bins=30)  # density=False would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data')

    plt.figure()
    plt.hist(msd_other1, density=True, bins=30)  # density=False would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data')

    plt.figure()
    plt.hist(msd_other2, density=True, bins=30)  # density=False would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data')

    plt.show()


if __name__ == '__main__':
    main()
