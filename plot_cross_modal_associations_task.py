"""Plot layer outputs of the model for the cross-modal associations task"""

import argparse
import random
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision.transforms import Lambda

import utils.checkpoint
from data.cross_modal_associations_dataset import CrossModalAssociationsDataset
from functions.autograd_functions import SpikeFunction
from functions.plasticity_functions import InvertedOjaWithSoftUpperBound
from models.network_models import CrossModalAssociations
from models.neuron_models import IafPscDelta
from models.protonet_models import SpikingProtoNet


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
    parser.add_argument('--num_time_steps', default=100, type=int, metavar='N',
                        help='Number of time steps for each item in the sequence (default: 100)')
    parser.add_argument('--fix_cnn_thresholds', action='store_false',
                        help='Do not adjust firing threshold after conversion (default: will adjust v_th via a bias)')

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
                        help='Spike threshold (default: 0.1)')
    parser.add_argument('--perfect_reset', action='store_true',
                        help='Set the membrane potential to zero after a spike')
    parser.add_argument('--refractory_time_steps', default=3, type=int, metavar='N',
                        help='The number of time steps the neuron is refractory (default: 3)')
    parser.add_argument('--tau_mem', default=20.0, type=float, metavar='N',
                        help='Neuron membrane time constant (default: 20.0)')

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
    image_embedding_layer = SpikingProtoNet(IafPscDelta(thr=args.thr,
                                                        perfect_reset=args.perfect_reset,
                                                        refractory_time_steps=args.refractory_time_steps,
                                                        tau_mem=args.tau_mem,
                                                        spike_function=SpikeFunction),
                                            input_size=784,
                                            output_size=args.embedding_size,
                                            num_time_steps=args.num_time_steps,
                                            refractory_time_steps=args.refractory_time_steps,
                                            use_bias=args.fix_cnn_thresholds)

    mfcc_embedding_layer = SpikingProtoNet(IafPscDelta(thr=args.thr,
                                                       perfect_reset=args.perfect_reset,
                                                       refractory_time_steps=args.refractory_time_steps,
                                                       tau_mem=args.tau_mem,
                                                       spike_function=SpikeFunction),
                                           input_size=600,
                                           output_size=args.embedding_size,
                                           num_time_steps=args.num_time_steps,
                                           refractory_time_steps=args.refractory_time_steps,
                                           use_bias=args.fix_cnn_thresholds)

    image_embedding_layer.threshold_balancing([1.8209, 11.5916, 4.1207, 2.6341])
    mfcc_embedding_layer.threshold_balancing([2.1098, 3.9579, 3.1872, 1.6841])

    # Create the model
    model = CrossModalAssociations(
        output_size=784,
        memory_size=args.memory_size,
        num_time_steps=args.num_time_steps,
        readout_delay=args.readout_delay,
        tau_trace=args.tau_trace,
        mfcc_embedding_layer=mfcc_embedding_layer,
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

    outputs, encoding_outputs, writing_outputs, reading_outputs, decoder_outputs = model(
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
    decoder_output_l1 = decoder_outputs[0].detach().numpy()
    decoder_output_l2 = decoder_outputs[1].detach().numpy()
    outputs = outputs.view(1, 28, 28).detach().numpy()

    mean_rate_mfcc_encoding = np.sum(mfcc_encoded, axis=1) / (1e-3 * args.sequence_length * args.num_time_steps)
    mean_rate_image_encoding = np.sum(images_encoded, axis=1) / (1e-3 * args.sequence_length * args.num_time_steps)
    mean_rate_query_encoding = np.sum(query_encoded, axis=1) / (1e-3 * args.num_time_steps)
    mean_rate_write_key = np.sum(write_key, axis=1) / (1e-3 * args.sequence_length * args.num_time_steps)
    mean_rate_write_val = np.sum(write_val, axis=1) / (1e-3 * args.sequence_length * args.num_time_steps)
    mean_rate_read_key = np.sum(read_key, axis=1) / (1e-3 * args.num_time_steps)
    mean_rate_read_val = np.sum(read_val, axis=1) / (1e-3 * args.num_time_steps)
    mean_rate_decoder_output_l1 = np.sum(decoder_output_l1, axis=1) / (1e-3 * args.num_time_steps)
    mean_rate_decoder_output_l2 = np.sum(decoder_output_l2, axis=1) / (1e-3 * args.num_time_steps)

    z_s_enc = np.concatenate((mfcc_encoded[0], images_encoded[0]), axis=1)
    z_r_enc = query_encoded[0]
    z_key = np.concatenate((write_key[0], read_key[0]), axis=0)
    z_value = np.concatenate((write_val[0], read_val[0]), axis=0)
    z_dec1 = decoder_output_l1[0]
    z_dec2 = decoder_output_l2[0]

    print("z_s_enc", z_s_enc.shape)
    print("z_r_enc", z_r_enc.shape)
    print("z_key", z_key.shape)
    print("z_value", z_value.shape)
    print("z_dec1", z_dec1.shape)
    print("z_dec2", z_dec2.shape)

    all_neurons = np.concatenate((
        np.pad(z_s_enc, ((0, args.num_time_steps), (0, 0))),
        np.pad(z_r_enc, ((z_s_enc.shape[0], 0), (0, 0))),
        z_key,
        z_value,
        np.pad(z_dec1, ((z_s_enc.shape[0], 0), (0, 0))),
        np.pad(z_dec2, ((z_s_enc.shape[0], 0), (0, 0))),
    ), axis=1)

    print("z_s_enc", (np.sum(z_s_enc, axis=0) / (1e-3 * (args.sequence_length + 1) * args.num_time_steps)).mean())
    print("z_r_enc", (np.sum(z_r_enc, axis=0) / (1e-3 * (args.sequence_length + 1) * args.num_time_steps)).mean())
    print("z_key", (np.sum(z_key, axis=0) / (1e-3 * (args.sequence_length + 1) * args.num_time_steps)).mean())
    print("z_value", (np.sum(z_value, axis=0) / (1e-3 * (args.sequence_length + 1) * args.num_time_steps)).mean())
    print("z_dec1", (np.sum(z_dec1, axis=0) / (1e-3 * (args.sequence_length + 1) * args.num_time_steps)).mean())
    print("z_dec2", (np.sum(z_dec2, axis=0) / (1e-3 * (args.sequence_length + 1) * args.num_time_steps)).mean())
    print("all_neurons", (np.sum(all_neurons, axis=0) / (1e-3 * (args.sequence_length + 1) * args.num_time_steps)).mean())

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

    fig, ax = plt.subplots(nrows=3, ncols=2, sharex='col', gridspec_kw={'width_ratios': [10, 1]})
    ax[0, 0].pcolormesh(mfcc_encoded[0].T, cmap='binary')
    ax[0, 0].set_ylabel('mfcc')
    ax[0, 1].barh(range(args.embedding_size), mean_rate_mfcc_encoding[0])
    ax[0, 1].set_ylim([0, args.embedding_size])
    ax[0, 1].set_yticks([])
    ax[1, 0].pcolormesh(images_encoded[0].T, cmap='binary')
    ax[1, 0].set_ylabel('images')
    ax[1, 1].barh(range(args.embedding_size), mean_rate_image_encoding[0])
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

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex='col', gridspec_kw={'width_ratios': [10, 1]})
    ax[0].pcolormesh(decoder_output_l1[0].T, cmap='binary')
    ax[0].set_ylabel('decoder layer 1')
    ax[1].barh(range(256), mean_rate_decoder_output_l1[0])
    ax[1].set_ylim([0, 256])
    ax[1].set_yticks([])

    fig, ax = plt.subplots(nrows=1, ncols=2, sharex='col', gridspec_kw={'width_ratios': [10, 1]})
    ax[0].pcolormesh(decoder_output_l2[0].T, cmap='binary')
    ax[0].set_ylabel('decoder layer 2')
    ax[1].barh(range(784), mean_rate_decoder_output_l2[0])
    ax[1].set_ylim([0, 784])
    ax[1].set_yticks([])

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex='all')
    ax.imshow(np.transpose(outputs, (1, 2, 0)), aspect='equal', cmap='gray', vmin=np.min(outputs), vmax=np.max(outputs))
    ax.set(title='Reconstructed image')
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
