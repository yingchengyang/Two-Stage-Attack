import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch
import torch.nn.functional as F
import time

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
from a2c_ppo_acktr import algo, utils
from evaluation import evaluate

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=True,
    help='whether to use a non-deterministic policy')
parser.add_argument('--device', default='cpu')
parser.add_argument('--attack_method', default='none', type=str)
parser.add_argument('--epsilon', default=0.0, type=float)
parser.add_argument('--target_attack', default=False, type=bool)
parser.add_argument('--target_path', default=None, type=str)


def run_attack(input, model, attack_method, epsilon, steps, target_model,
               recurrent_hidden_states, masks):
    assert attack_method in ('none', 'random', 'untarget_kl_fgsm',
                             'untarget_kl_pgd', 'target_kl_fgsm', 'target_kl_pgd',
                             'untarget_pi_fgsm', 'untarget_pi_pgd', 'target_pi_fgsm',
                             'target_pi_pgd'), 'Invalid attack type'
    # print(input.shape) # tensor [1, 4, 84, 84]
    # print(input) # no pre-process 0-255
    if attack_method == 'none':
        return input
    elif attack_method == 'random':
        noise = torch.empty_like(input).uniform_(epsilon, epsilon)
        return input + noise
    elif attack_method == 'untarget_kl_fgsm':
        with torch.no_grad():
            value, actor_features, rnn_hxs = model.base(input, recurrent_hidden_states,
                                                        masks)
            dist = model.dist(actor_features)
            original_output = dist.probs # print probs of each actio

        model.zero_grad()
        input = input.clone().detach().requires_grad_()
        value, actor_features, rnn_hxs = model.base(input, recurrent_hidden_states,
                                                    masks)
        dist = model.dist(actor_features)
        output = dist.probs  # print probs of each actio


        loss = F.kl_div(output, original_output, reduction='sum')
        # print(output)
        # print(original_output)
        # loss = ce(output, original_output)
        loss.backward()
        # l_infty
        # input = input + torch.sign(input.grad) * epsilon
        # l_2
        # print(torch.norm(input.grad))
        input = input + input.grad / torch.norm(input.grad) * epsilon * 84.0 * 2 * 255
        return input
    elif attack_method == 'target_kl_fgsm':
        with torch.no_grad():
            target_model.zero_grad()
            value, actor_features, rnn_hxs = target_model.base(input, recurrent_hidden_states,
                                                        masks)
            dist = model.dist(actor_features)
            target_output = dist.probs  # print probs of each actio

        model.zero_grad()
        input = input.clone().detach().requires_grad_()
        value, actor_features, rnn_hxs = model.base(input, recurrent_hidden_states,
                                                    masks)
        dist = model.dist(actor_features)
        output = dist.probs  # print probs of each actio

        loss = F.kl_div(output, target_output, reduction='sum')
        loss.backward()
        # l_infty
        # input = input - torch.sign(input.grad) * epsilon
        # l_2
        input = input - input.grad / torch.norm(input.grad) * epsilon * 84.0 * 2 * 255

        return input
    elif attack_method == 'untarget_kl_pgd':
        with torch.no_grad():
            value, actor_features, rnn_hxs = model.base(input, recurrent_hidden_states,
                                                        masks)
            dist = model.dist(actor_features)
            original_output = dist.probs  # print probs of each actio

            epsilon_step = epsilon / steps

        for _ in range(steps):
            model.zero_grad()
            input = input.clone().detach().requires_grad_()

            value, actor_features, rnn_hxs = model.base(input, recurrent_hidden_states,
                                                        masks)
            dist = model.dist(actor_features)
            output = dist.probs  # print probs of each actio

            loss = F.kl_div(output, original_output, reduction='sum')
            loss.backward()
            # l_infty
            # input = input + torch.sign(input.grad) * epsilon_step
            # l_2
            input = input + input.grad / torch.norm(input.grad) * epsilon_step * 84.0 * 2 * 255
            # print(input)
        return input
    elif attack_method == 'target_kl_pgd':
        with torch.no_grad():
            target_model.zero_grad()
            value, actor_features, rnn_hxs = target_model.base(input, recurrent_hidden_states,
                                                               masks)
            dist = model.dist(actor_features)
            target_output = dist.probs  # print probs of each actio

            epsilon_step = epsilon / steps

        for _ in range(steps):
            model.zero_grad()
            input = input.clone().detach().requires_grad_()

            value, actor_features, rnn_hxs = model.base(input, recurrent_hidden_states,
                                                        masks)
            dist = model.dist(actor_features)
            output = dist.probs  # print probs of each actio

            loss = F.kl_div(output, target_output, reduction='sum')
            loss.backward()

            # l_infty
            # input = input - torch.sign(input.grad) * epsilon_step
            # l_2
            input = input - input.grad / torch.norm(input.grad) * epsilon_step * 84.0 * 2 * 255
        return input
    return input


def main(args):
    args.det = not args.non_det
    env = make_vec_envs(args.env_name, args.seed, 1,
                        0.99, '/tmp/gym/', args.device, False)
    print(env.action_space)

    # We need to use the same statistics for normalization as used in training
    actor_critic, obs_rms = torch.load(os.path.join(args.load_dir),
                                       map_location=args.device)

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    recurrent_hidden_states = torch.zeros(1,
                                          actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    obs = env.reset()

    target_actor_critic = None
    if args.target_attack:
        target_actor_critic, _ = torch.load(os.path.join(args.target_path),
                                            map_location=args.device)
    eval_episode_rewards = []
    start_time = time.time()
    while len(eval_episode_rewards) < 5:
        obs = run_attack(obs, actor_critic, args.attack_method, args.epsilon,
                         10, target_actor_critic, recurrent_hidden_states, masks)
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

            # action: e.g. tensor([[0]]), tensor([[2]])
            # print(action)
            # value: e.g. tensor([[1.5200]])
            # print(value)

        # Obser reward and next obs
        # print(obs)
        obs, reward, done, infos = env.step(action)
        masks.fill_(0.0 if done else 1.0)

        if done:
            # print(total_rewards)
            # total_rewards = 0.0
            obs = env.reset()
            # print('infos',infos)
            for info in infos:
                if 'episode' in info.keys():
                    print('This is episode ', len(eval_episode_rewards)+1)
                    print('reward of episode', info['episode']['r'])
                    print('time cost:', time.time() - start_time)
                    start_time = time.time()
                    eval_episode_rewards.append(info['episode']['r'])

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
            len(eval_episode_rewards), np.mean(eval_episode_rewards)))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)