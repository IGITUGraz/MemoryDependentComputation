"""Plot layer outputs of the model for the memorizing associations tasks"""

import argparse
import random
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils.checkpoint import load_checkpoint
from data.memorizing_associations_dataset import MemorizingAssociationsDataset
from functions.autograd_functions import SpikeFunction
from functions.plasticity_functions import InvertedOjaWithSoftUpperBound
from models.network_models import MemorizingAssociations
from models.neuron_models import IafPscDelta


def main():
    parser = argparse.ArgumentParser(description='Memorizing associations task plotting')
    parser.add_argument('--checkpoint_path', default='', type=str, metavar='PATH',
                        help='Path to checkpoint (default: none)')
    parser.add_argument('--check_params', default=1, type=int, choices=[0, 1], metavar='CHECK_PARAMS',
                        help='When loading from a checkpoint check if the model was trained with the same parameters '
                             'as requested now (default: 1)')

    parser.add_argument('--sequence_length', default=10, type=int, metavar='SEQUENCE_LENGTH',
                        help='The number of vector-label pairs (default: 10)')
    parser.add_argument('--num_classes', default=10, type=int, metavar='NUM_CLASSES',
                        help='The number of classes (default: 10)')
    parser.add_argument('--feature_size', default=10, type=int, metavar='FEATURE_SIZE',
                        help='Size of the input features (default: 10)')
    parser.add_argument('--num_time_steps', default=100, type=int, metavar='N',
                        help='Number of time steps for each fact (default: 100)')

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
    parser.add_argument('--readout_delay', default=1, type=int, metavar='N',
                        help='Synaptic delay of the feedback-connections from value-neurons to key-neurons in the '
                             'reading layer (default: 1)')

    parser.add_argument('--thr', default=0.1, type=float, metavar='N',
                        help='Spike threshold (default: 0.1)')
    parser.add_argument('--perfect_reset', action='store_true',
                        help='Set the membrane potential to zero after a spike')
    parser.add_argument('--refractory_time_steps', default=3, type=int, metavar='N',
                        help='The number of time steps the neuron is refractory (default: 3)')
    parser.add_argument('--tau_mem', default=20.0, type=float, metavar='N',
                        help='Neuron membrane time constant (default: 20.0)')

    parser.add_argument('--dataset_seed', default=42, type=int, metavar='N',
                        help='Seed for creating the dataset (default: 42)')
    parser.add_argument('--seed', default=None, type=int, metavar='N',
                        help='Seed for initializing (default: none)')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading code
    dataset = MemorizingAssociationsDataset(sequence_length=args.sequence_length, num_classes=args.num_classes,
                                            feature_size=args.feature_size, inf_data=False, dataset_size=1,
                                            seed=args.dataset_seed)

    # Create the model
    model = MemorizingAssociations(
        input_size=args.feature_size,
        output_size=args.num_classes,
        num_embeddings=args.num_classes,
        embedding_size=args.embedding_size,
        memory_size=args.memory_size,
        num_time_steps=args.num_time_steps,
        readout_delay=args.readout_delay,
        tau_trace=args.tau_trace,
        plasticity_rule=InvertedOjaWithSoftUpperBound(w_max=args.w_max,
                                                      gamma_pos=args.gamma_pos,
                                                      gamma_neg=args.gamma_neg),
        dynamics=IafPscDelta(thr=args.thr,
                             perfect_reset=args.perfect_reset,
                             refractory_time_steps=args.refractory_time_steps,
                             tau_mem=args.tau_mem,
                             spike_function=SpikeFunction))

    # Load checkpoint
    if args.checkpoint_path:
        print("=> loading checkpoint '{}'".format(args.checkpoint_path))
        checkpoint = load_checkpoint(args.checkpoint_path, device)
        best_acc = checkpoint['best_acc']
        print("Best accuracy {}".format(best_acc))
        if args.check_params:
            for key, val in vars(args).items():
                if key not in ['check_params', 'seed', 'dataset_seed', 'checkpoint_path', 'example']:
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
    sample, sequence_length = dataset[0]
    features = sample['features'].unsqueeze(0)
    labels = sample['labels'].unsqueeze(0)
    query = sample['query'].unsqueeze(0)

    outputs, encoding_outputs, writing_outputs, reading_outputs = model(features, labels, query)

    # Get the outputs
    features_encoded = encoding_outputs[0].detach().numpy()
    labels_encoded = encoding_outputs[1].detach().numpy()
    query_encoded = encoding_outputs[2].detach().numpy()
    mem = writing_outputs[0].detach().numpy()
    write_key = writing_outputs[1].detach().numpy()
    write_val = writing_outputs[2].detach().numpy()
    read_key = reading_outputs[0].detach().numpy()
    read_val = reading_outputs[1].detach().numpy()
    outputs = outputs.detach().numpy()

    mean_rate_features_encoding = np.sum(features_encoded, axis=1) / (1e-3 * sequence_length * args.num_time_steps)
    mean_rate_labels_encoding = np.sum(labels_encoded, axis=1) / (1e-3 * sequence_length * args.num_time_steps)
    mean_rate_query_encoding = np.sum(query_encoded, axis=1) / (1e-3 * args.num_time_steps)
    mean_rate_write_key = np.sum(write_key, axis=1) / (1e-3 * sequence_length * args.num_time_steps)
    mean_rate_write_val = np.sum(write_val, axis=1) / (1e-3 * sequence_length * args.num_time_steps)
    mean_rate_read_key = np.sum(read_key, axis=1) / (1e-3 * args.num_time_steps)
    mean_rate_read_val = np.sum(read_val, axis=1) / (1e-3 * args.num_time_steps)

    # Make some plots
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex='all')
    ax[0].pcolormesh(features[0].T, cmap='binary')
    ax[0].set_ylabel('features')
    ax[1].pcolormesh(labels[0, np.newaxis], cmap='binary')
    ax[1].set_ylabel('labels')
    ax[2].pcolormesh(query.T, cmap='binary')
    ax[2].set_ylabel('query')
    plt.tight_layout()

    fig, ax = plt.subplots(nrows=3, ncols=2, sharex='col', gridspec_kw={'width_ratios': [10, 1]})
    ax[0, 0].pcolormesh(features_encoded[0].T, cmap='binary')
    ax[0, 0].set_ylabel('features')
    ax[0, 1].barh(range(args.embedding_size), mean_rate_features_encoding[0])
    ax[0, 1].set_ylim([0, args.embedding_size])
    ax[0, 1].set_yticks([])

    ax[1, 0].pcolormesh(labels_encoded[0].T, cmap='binary')
    ax[1, 0].set_ylabel('labels')
    ax[1, 1].barh(range(args.embedding_size), mean_rate_labels_encoding[0])
    ax[1, 1].set_ylim([0, args.embedding_size])
    ax[1, 1].set_yticks([])

    ax[2, 0].pcolormesh(query_encoded[0].T, cmap='binary')
    ax[2, 0].set_ylabel('query')
    ax[2, 1].barh(range(args.embedding_size), mean_rate_query_encoding[0])
    ax[2, 1].set_ylim([0, args.embedding_size])
    ax[2, 1].set_yticks([])
    plt.tight_layout()

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='col', gridspec_kw={'width_ratios': [10, 1]})
    ax[0, 0].pcolormesh(write_key[0].T, cmap='binary')
    ax[0, 0].set_ylabel('write keys')
    ax[0, 1].barh(range(args.memory_size), mean_rate_write_key[0])
    ax[0, 1].set_ylim([0, args.memory_size])
    ax[0, 1].set_yticks([])
    ax[1, 0].pcolormesh(write_val[0].T, cmap='binary')
    ax[1, 0].set_ylabel('write values')
    ax[1, 1].barh(range(args.memory_size), mean_rate_write_val[0])
    ax[1, 1].set_ylim([0, args.memory_size])
    ax[1, 1].set_yticks([])
    plt.tight_layout()

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='all')
    ax.matshow(mem[0], cmap='RdBu')
    plt.tight_layout()

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex='col', gridspec_kw={'width_ratios': [10, 1]})
    ax[0, 0].pcolormesh(read_key[0].T, cmap='binary')
    ax[0, 0].set_ylabel('read keys')
    ax[0, 1].barh(range(args.memory_size), mean_rate_read_key[0])
    ax[0, 1].set_ylim([0, args.memory_size])
    ax[0, 1].set_yticks([])
    ax[1, 0].pcolormesh(read_val[0].T, cmap='binary')
    ax[1, 0].set_ylabel('read values')
    ax[1, 1].barh(range(args.memory_size), mean_rate_read_val[0])
    ax[1, 1].set_ylim([0, args.memory_size])
    ax[1, 1].set_yticks([])
    plt.tight_layout()

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='all')
    ax.pcolormesh(outputs[0, None].T, cmap='binary')
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
