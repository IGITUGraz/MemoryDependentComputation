"""Plot layer outputs of the model for the Omniglot one-shot task"""

import argparse
import random
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

import utils.checkpoint
from data.omniglot_dataset import OmniglotDataset
from functions.autograd_functions import SpikeFunction
from functions.plasticity_functions import InvertedOjaWithSoftUpperBound
from models.network_models import OmniglotOneShot
from models.neuron_models import IafPscDelta
from models.protonet_models import SpikingProtoNet
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
    parser.add_argument('--num_time_steps', default=100, type=int, metavar='N',
                        help='Number of time steps for each item in the sequence (default: 100)')

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
    parser.add_argument('--fix_cnn_thresholds', action='store_false',
                        help='Do not adjust firing threshold after conversion'
                             '(default: will adjust v_th via a bias)')

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
    image_embedding_layer = SpikingProtoNet(IafPscDelta(thr=args.thr,
                                                        perfect_reset=args.perfect_reset,
                                                        refractory_time_steps=args.refractory_time_steps,
                                                        tau_mem=args.tau_mem,
                                                        spike_function=SpikeFunction),
                                            input_size=784,
                                            num_time_steps=args.num_time_steps,
                                            refractory_time_steps=args.refractory_time_steps,
                                            use_bias=args.fix_cnn_thresholds)
    image_embedding_layer.threshold_balancing([0.9151, 4.7120, 3.1376, 2.6661])

    # Create the model
    model = OmniglotOneShot(
        num_embeddings=args.num_classes,
        output_size=args.num_classes,
        memory_size=args.memory_size,
        num_time_steps=args.num_time_steps,
        readout_delay=args.readout_delay,
        tau_trace=args.tau_trace,
        image_embedding_layer=image_embedding_layer,
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

    mean_rate_facts_encoding = np.sum(facts_encoded, axis=1) / (1e-3 * args.num_classes * args.num_time_steps)
    mean_rate_labels_encoding = np.sum(labels_encoded, axis=1) / (1e-3 * args.num_classes * args.num_time_steps)
    mean_rate_query_encoding = np.sum(query_encoded, axis=1) / (1e-3 * args.num_time_steps)
    mean_rate_write_key = np.sum(write_key, axis=1) / (1e-3 * args.num_classes * args.num_time_steps)
    mean_rate_write_val = np.sum(write_val, axis=1) / (1e-3 * args.num_classes * args.num_time_steps)
    mean_rate_read_key = np.sum(read_key, axis=1) / (1e-3 * args.num_time_steps)
    mean_rate_read_val = np.sum(read_val, axis=1) / (1e-3 * args.num_time_steps)

    z_s_enc = np.concatenate((labels_encoded[0], facts_encoded[0]), axis=1)
    z_r_enc = query_encoded[0]
    z_key = np.concatenate((write_key[0], read_key[0]), axis=0)
    z_value = np.concatenate((write_val[0], read_val[0]), axis=0)

    print("z_s_enc", z_s_enc.shape)
    print("z_r_enc", z_r_enc.shape)
    print("z_key", z_key.shape)
    print("z_value", z_value.shape)

    all_neurons = np.concatenate((
        np.pad(z_s_enc, ((0, args.num_time_steps), (0, 0))),
        np.pad(z_r_enc, ((z_s_enc.shape[0], 0), (0, 0))),
        z_key,
        z_value
    ), axis=1)

    print("z_s_enc", (np.sum(z_s_enc, axis=0) / (1e-3 * (args.num_classes + 1) * args.num_time_steps)).mean())
    print("z_r_enc", (np.sum(z_r_enc, axis=0) / (1e-3 * (args.num_classes + 1) * args.num_time_steps)).mean())
    print("z_key", (np.sum(z_key, axis=0) / (1e-3 * (args.num_classes + 1) * args.num_time_steps)).mean())
    print("z_value", (np.sum(z_value, axis=0) / (1e-3 * (args.num_classes + 1) * args.num_time_steps)).mean())
    print("all_neurons", (np.sum(all_neurons, axis=0) / (1e-3 * (args.num_classes + 1) * args.num_time_steps)).mean())

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

    fig, ax = plt.subplots(nrows=3, ncols=2, sharex='col', gridspec_kw={'width_ratios': [10, 1]})
    ax[0, 0].pcolormesh(facts_encoded[0].T, cmap='binary')
    ax[0, 0].set_ylabel('facts')
    ax[0, 1].barh(range(model.images_embedding_layer.output_size), mean_rate_facts_encoding[0])
    ax[0, 1].set_ylim([0, model.images_embedding_layer.output_size])
    ax[0, 1].set_yticks([])
    ax[1, 0].pcolormesh(labels_encoded[0].T, cmap='binary')
    ax[1, 0].set_ylabel('labels')
    ax[1, 1].barh(range(model.images_embedding_layer.output_size), mean_rate_labels_encoding[0])
    ax[1, 1].set_ylim([0, model.images_embedding_layer.output_size])
    ax[1, 1].set_yticks([])
    ax[2, 0].pcolormesh(query_encoded[0].T, cmap='binary')
    ax[2, 0].set_ylabel('query')
    ax[2, 1].barh(range(model.images_embedding_layer.output_size), mean_rate_query_encoding[0])
    ax[2, 1].set_ylim([0, model.images_embedding_layer.output_size])
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

    test_sampler = EpisodicBatchSampler(test_set.y, num_classes=args.num_classes, batch_size=1, iterations=1,
                                        seed=args.sampler_seed)
    test_loader = torch.utils.data.DataLoader(test_set, batch_sampler=test_sampler)

    mems = []
    for i in range(100):
        x, y = next(iter(test_loader))
        x = x.view(1, -1, *x.size()[1:])

        facts = x[:, :args.num_classes]
        query = x[:, -1]

        labels = torch.arange(0, args.num_classes)
        labels = labels.expand(1, args.num_classes).long()

        _, _, writing_outputs, _ = model(facts, labels, query)
        mem = writing_outputs[0].squeeze(0).detach().numpy()
        mems.append(mem.flatten())

    mem_values = np.concatenate(mems)

    plt.figure()
    sns.histplot(mem_values, kde=True, bins='auto', log_scale=[False, True])
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
