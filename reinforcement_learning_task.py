import argparse
import collections
import json
import os
import random
import socket
import sys
import warnings
from datetime import datetime
from typing import List, Union, Optional

import numpy as np
import torch
import torch.nn.functional
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils.checkpoint
from agents.ppo import PPO
from environments.concentration import Concentration
from environments.multiprocessing_environment import make_vec_envs, PyTorchVecEnv
from functions.autograd_functions import SpikeFunction
from functions.plasticity_functions import InvertedOjaWithSoftUpperBound
from models.network_models import ReinforcementLearning
from models.neuron_models import IafPscDelta
from policies.policy import Policy
from storage.rollout_storage import RolloutStorage

parser = argparse.ArgumentParser(description='Reinforcement learning task training')
parser.add_argument('--num_envs', default=64, type=int, metavar='NUM_ENVS',
                    help='Number of parallel environments (default: 64)')
parser.add_argument('--num_cells', default=4, type=int, metavar='NUM_CELLS',
                    help='Number of cells (default: 4)')
parser.add_argument('--card_shape', default=10, type=int, metavar='CARD_SHAPE',
                    help='Shape of each card on the grid (default: 10)')
parser.add_argument('--resample_cards', action='store_true',
                    help='Sample a new deck of cards for each game.')
parser.add_argument('--match_reward', default=25.0, type=float, metavar='MATCH_REWARD',
                    help='Reward the agent receives when it has flipped two matching cards (default: 25.0)')
parser.add_argument('--flip_penalty', default=-0.5, type=float, metavar='FLIP_PENALTY',
                    help='Penalty for each card the agent flips (default: -0.5)')
parser.add_argument('--num_time_steps', default=100, type=int, metavar='N',
                    help='Number of time steps for each observation (default: 100)')

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

parser.add_argument('--use_rms_prop', action='store_true',
                    help='Use RMSprop optimizer instead of Adam (only for A2C)')
parser.add_argument('--rms_prop_eps', default=1e-5, type=float, metavar='N',
                    help='Epsilon parameter of RMSprop (default: 1e-5)')
parser.add_argument('--rms_prop_alpha', default=0.99, type=float, metavar='N',
                    help='Alpha parameter of RMSprop (default: 0.99)')
parser.add_argument('--adam_eps', default=1e-5, type=float, metavar='N',
                    help='Epsilon parameter of Adam (default: 1e-5)')
parser.add_argument('--num_env_steps', default=100e6, type=int, metavar='N',
                    help='Number of environment steps to train (default: 100e6)')
parser.add_argument('--num_steps', default=10, type=int, metavar='NUM_STEPS',
                    help='The number of forward steps (default: 10)')
parser.add_argument('--learning_rate', default=3e-4, type=float, metavar='N',
                    help='Initial learning rate (default: 3e-4)')
parser.add_argument('--decay_lr_linearly', action='store_true',
                    help='Use a linearly decaying learning rate')
parser.add_argument('--learning_rate_decay', default=1.0, type=float, metavar='N',
                    help='Learning rate decay if not using linear decay (default: 1.0)')
parser.add_argument('--decay_learning_rate_every', default=2000, type=int, metavar='N',
                    help='Decay the learning rate every N updates (default: 2000)')
parser.add_argument('--discount_factor', default=0.99, type=float, metavar='N',
                    help='Factor for calculating discounted rewards (default: 0.99)')
parser.add_argument('--ppo_clip_param', default=0.2, type=float, metavar='N',
                    help='PPO clip parameter (default: 0.2)')
parser.add_argument('--ppo_batches', default=16, type=int, metavar='N',
                    help='Number of PPO batches (default: 16)')
parser.add_argument('--ppo_epochs', default=4, type=int, metavar='N',
                    help='Number of PPO epochs (default: 4)')
parser.add_argument('--value_coefficient', default=0.1, type=float, metavar='N',
                    help='Value loss weight factor (default: 0.1)')
parser.add_argument('--entropy_coefficient', default=0.01, type=float, metavar='N',
                    help='Entropy loss weight factor (default: 0.01)')
parser.add_argument('--normalize_advantage', action='store_true',
                    help='Whether to normalize the advantage in A2C or not')
parser.add_argument('--use_smooth_l1_loss', action='store_true',
                    help='Use smooth L1 for the value loss instead of MSE loss in A2C')
parser.add_argument('--max_grad_norm', default=0.5, type=float, metavar='N',
                    help='Gradients with an L2 norm larger than max_grad_norm will be clipped '
                         'to have an L2 norm of max_grad_norm. If non-positive, then the gradient will '
                         'not be clipped. (default: 0.5)')

parser.add_argument('--start_update', default=0, type=int, metavar='N',
                    help='Manual update number (useful on restarts, default: 0)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', action='store_true',
                    help='Evaluate the model')

parser.add_argument('--logging', action='store_true',
                    help='Write tensorboard logs')
parser.add_argument('--cuda', action='store_true',
                    help='CUDA flag')
parser.add_argument('--seed', default=None, type=int, metavar='N',
                    help='Seed for initializing training (default: None)')
parser.add_argument('--env_seed', default=42, type=int, metavar='N',
                    help='Base seed for environments (default: 42)')


def main():
    writer = None
    log_dir = None
    time_stamp = datetime.now().strftime('%b%d_%H-%M-%S')

    with open('version.txt') as f:
        version = f.readline()

    args = parser.parse_args()

    args.num_env_steps = int(args.num_env_steps)
    args.use_rms_prop = True if args.use_rms_prop else False
    args.resample_cards = True if args.resample_cards else False
    args.decay_lr_linearly = True if args.decay_lr_linearly else False
    args.use_smooth_l1_loss = True if args.use_smooth_l1_loss else False
    args.normalize_advantage = True if args.normalize_advantage else False
    args.max_grad_norm = args.max_grad_norm if args.max_grad_norm > 0 else None
    args.evaluate = True if args.evaluate else False
    args.logging = True if args.logging else False
    args.cuda = True if args.cuda else False

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can '
                      'slow down your training considerably! You may see unexpected behavior when restarting from '
                      'checkpoints.')

    if torch.cuda.is_available() and args.cuda:
        args.cuda = True
    else:
        args.cuda = False
        print('Using CPU ...')
    device = torch.device("cuda" if args.cuda else "cpu")

    # Environments
    envs = make_vec_envs(Concentration(num_cells=args.num_cells,
                                       card_shape=args.card_shape,
                                       match_reward=args.match_reward,
                                       flip_penalty=args.flip_penalty,
                                       resample_cards=args.resample_cards),
                         args.num_envs, args.env_seed, device)

    # Actor-Critic
    actor_critic = Policy(
        action_space=envs.action_space,
        net=ReinforcementLearning(input_size=envs.observation_space.shape[0],
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
    ).to(device)

    # Agent
    agent = PPO(actor_critic=actor_critic,
                value_coefficient=args.value_coefficient,
                entropy_coefficient=args.entropy_coefficient,
                max_grad_norm=args.max_grad_norm,
                learning_rate=args.learning_rate,
                ppo_clip_param=args.ppo_clip_param,
                ppo_batches=args.ppo_batches,
                ppo_epochs=args.ppo_epochs,
                adam_eps=args.adam_eps)

    # Rollout storage
    rollouts = RolloutStorage(num_steps=args.num_steps,
                              num_environments=args.num_envs,
                              observations_size=envs.observation_space.shape,
                              action_space=envs.action_space,
                              recurrent_state_size=actor_critic.state_size)

    # Optionally load from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = utils.checkpoint.load_checkpoint(args.resume, device)

            args.start_update = checkpoint['update']
            log_dir = checkpoint['log_dir']
            time_stamp = checkpoint['time_stamp']

            ignore_keys = ['num_env_steps', 'start_update', 'resume', 'cuda', 'evaluate', 'logging']

            if args.evaluate:
                ignore_keys += ['num_envs', 'env_seed']

            for key, val in vars(checkpoint['params']).items():
                if key not in ignore_keys:
                    if vars(args)[key] != val:
                        print("=> You tried to restart training of a model that was trained with different parameters "
                              "as you requested now. Aborting...")
                        sys.exit()

            actor_critic.load_state_dict(checkpoint['state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['update']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    suffix = f'num_envs{args.num_envs}-num_cells{args.num_cells}'

    # Summary writer
    if log_dir and args.logging and not args.evaluate:
        # Use the directory that is stored in checkpoint if we resume training
        writer = SummaryWriter(log_dir=log_dir)
    elif args.logging and not args.evaluate:
        log_dir = os.path.join('results', 'runs', time_stamp + '_' + socket.gethostname() +
                               f'_version-{version}-{suffix}_reinforcement-learning-task')
        writer = SummaryWriter(log_dir=log_dir)

        def pretty_json(hp):
            json_hp = json.dumps(hp, indent=2, sort_keys=False)
            return "".join('\t' + line for line in json_hp.splitlines(True))

        writer.add_text('Info/params', pretty_json(vars(args)))

    if args.evaluate:
        evaluate(actor_critic, args, device)
    else:
        train(actor_critic, agent, envs, rollouts, args, writer, time_stamp, version, suffix, device)


def train(actor_critic: torch.nn.Module, agent: callable, envs: PyTorchVecEnv, rollouts: RolloutStorage,
          args: argparse.Namespace, writer: Union[None, SummaryWriter], time_stamp: str, version: str, suffix: str,
          device: Union[str, torch.device]):
    actor_critic.train()

    observation = envs.reset()
    rollouts.observations[0].copy_(observation)
    rollouts.to(device)

    steps_until_done = collections.deque(maxlen=30)

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_envs
    progress = tqdm(range(args.start_update, num_updates))

    num_games = 0
    min_mean_steps = np.inf
    for update in progress:

        current_lr = adjust_learning_rate(agent.optimizer, update, num_updates, args)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_states = actor_critic.get_action(
                    inputs=rollouts.observations[step],
                    states=[state[step] for state in rollouts.recurrent_states],
                    mask=rollouts.masks[step])

            # Observe reward and next obs
            observation, reward, done, infos = envs.step(action)

            for info, done_ in zip(infos, done):
                if done_:
                    steps_until_done.append(info['num_steps'])

            num_games += np.sum([1 if done_ else 0 for done_ in done])

            # If done then clean the history of observations.
            mask = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_mask = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(observation=observation,
                            action_log_prob=action_log_prob,
                            action=action,
                            value=value,
                            reward=reward,
                            mask=mask,
                            bad_mask=bad_mask,
                            recurrent_states=recurrent_states)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                inputs=rollouts.observations[-1],
                states=[state[-1] for state in rollouts.recurrent_states],
                mask=rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.discount_factor, False)

        loss, value_loss, policy_loss, entropy, var_explained = agent.update(rollouts)

        rollouts.after_update()

        mean_steps = np.mean(steps_until_done) if steps_until_done else np.nan
        min_steps = np.min(steps_until_done) if steps_until_done else np.nan
        max_steps = np.max(steps_until_done) if steps_until_done else np.nan

        is_best = mean_steps < min_mean_steps
        min_mean_steps = min(mean_steps, min_mean_steps)

        # Save checkpoint
        save_checkpoint(update, actor_critic.state_dict(), agent.optimizer.state_dict(), args, list(steps_until_done),
                        writer, time_stamp, version, suffix, is_best)

        progress.set_description(f'Update {update + 1}/{num_updates} | '
                                 f'Total steps: {(update + 1) * args.num_envs * args.num_steps} | '
                                 f'Loss: {loss:.4f} | '
                                 f'V-loss: {value_loss:.4f} | '
                                 f'P-loss: {policy_loss:.4f} | '
                                 f'Entropy: {policy_loss:.4f} | '
                                 f'Steps mean (min,max) {mean_steps:.2f} ({min_steps},{max_steps}) ')

        if writer is not None:
            writer.add_scalar('Loss/a2c', loss, update)
            writer.add_scalar('Loss/value', args.value_coefficient * value_loss, update)
            writer.add_scalar('Loss/policy', policy_loss, update)
            writer.add_scalar('Loss/entropy', args.entropy_coefficient * entropy, update)
            writer.add_scalar("Info/steps", mean_steps, update)
            writer.add_scalar("Info/games", num_games, update)
            writer.add_scalar("Info/steps4", steps_until_done.count(4), update)
            writer.add_scalar("Info/steps6", steps_until_done.count(6), update)
            writer.add_scalar("Info/steps8", steps_until_done.count(8), update)
            writer.add_scalar("Info/steps10", steps_until_done.count(10), update)
            if update == 0:
                writer.add_custom_scalars_multilinechart(["Info/steps4", "Info/steps6", "Info/steps8", "Info/steps10"])
            writer.add_scalar("Info/mean_returns", rollouts.returns.mean().item(), update)
            writer.add_scalar("Info/variance_explained", var_explained, update)
            writer.add_scalar("Info/learning_rate", current_lr, update)

    envs.close()


def evaluate(actor_critic: torch.nn.Module, args: argparse.Namespace, device: Union[str, torch.device],
             deterministic: Optional[bool] = True, min_num_games_to_finish: Optional[int] = 1000,
             max_steps: Optional[int] = 10000) -> None:

    envs = make_vec_envs(Concentration(num_cells=args.num_cells,
                                       card_shape=args.card_shape,
                                       match_reward=args.match_reward,
                                       flip_penalty=args.flip_penalty,
                                       resample_cards=args.resample_cards),
                         args.num_envs, args.env_seed + 1, device)

    observations = envs.reset()
    recurrent_states = [torch.zeros(args.num_envs, *size, device=device) for size in actor_critic.state_size]
    mask = torch.zeros(args.num_envs, 1, device=device)

    actor_critic.eval()

    step = 0
    steps_until_done = []
    while len(steps_until_done) < min_num_games_to_finish and step < max_steps:
        with torch.no_grad():
            _, action, _, recurrent_states, _ = actor_critic.get_action(
                inputs=observations,
                states=recurrent_states,
                mask=mask,
                deterministic=deterministic)

        # Observe reward and next obs
        observations, _, done, infos = envs.step(action)

        for info, done_ in zip(infos, done):
            if done_:
                steps_until_done.append(info['num_steps'])

        mask = torch.tensor([[0.0] if done_ else [1.0] for done_ in done], dtype=torch.float32, device=device)
        step += 1

    envs.close()

    if steps_until_done:
        steps, counts = np.unique(steps_until_done, return_counts=True)
        print("\nEvaluation using {} episodes: Mean steps {:.2f} (min steps {}, max steps {})\n"
              "Steps: {}, Counts: {}".format(len(steps_until_done), np.mean(steps_until_done), np.min(steps_until_done),
                                             np.max(steps_until_done), steps, counts))
    else:
        print("\nEvaluation did not finish within {} steps".format(max_steps))


def adjust_learning_rate(optimizer: torch.optim.Optimizer, update: int, total_updates: int,
                         args: argparse.Namespace) -> float:
    if args.decay_lr_linearly:
        """Decreases the learning rate linearly"""
        lr = args.learning_rate - (args.learning_rate * (update / float(total_updates)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        """Sets the learning rate to the initial LR decayed by X% every N epochs"""
        lr = args.learning_rate * (args.learning_rate_decay ** (update // args.decay_learning_rate_every))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr


def save_checkpoint(update: int, actor_critic_state_dict: dict, optimizer_state_dict: dict, args: argparse.Namespace,
                    steps_until_done: List[int], writer: Union[None, SummaryWriter], time_stamp: str, version: str,
                    suffix: str, is_best: bool) -> None:
    utils.checkpoint.save_checkpoint({
        'params': args,
        'update': update,
        'steps_until_done': steps_until_done,
        'state_dict': actor_critic_state_dict,
        'optimizer': optimizer_state_dict,
        'time_stamp': time_stamp,
        'log_dir': writer.get_logdir() if writer is not None else '',
    }, is_best, filename=os.path.join(
        'results', 'checkpoints', time_stamp +
        '_' + socket.gethostname() + f'_version-{version}-{suffix}_reinforcement-learning-task'))


if __name__ == '__main__':
    main()
