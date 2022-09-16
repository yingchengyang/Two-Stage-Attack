import argparse
import math
import os
import random
import time

import gym
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import A3Cff
from environment import atari_env
from utils import read_config

from torch.autograd import Variable

from adv_attacks.adv_model import PytorchModel


def get_parser():
    parser = argparse.ArgumentParser(description='A3C')

    parser.add_argument(
        '--max-episode-length',
        type=int,
        default=10000,
        metavar='M',
        help='maximum length of an episode (default: 10000)')
    parser.add_argument(
        '--env',
        default='PongNoFrameskip-v4',
        metavar='ENV',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--env-config',
        default='config.json',
        metavar='EC',
        help='environment to crop and resize info (default: config.json)')
    parser.add_argument(
        '--load-path',
        default='trained_models/PongNoFrameskip-v4_robust.pt',
        metavar='LMD',
        help='folder to load trained models from')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=-1,
        help='GPU to use [-1 CPU only] (default: -1)')
    parser.add_argument(
        '--skip-rate',
        type=int,
        default=4,
        metavar='SR',
        help='frame skip rate (default: 4)')
    parser.add_argument(
        '--fgsm-video',
        type=float,
        default=None,
        metavar='FV',
        help='whether to to produce a video of the agent performing under FGSM attack with given epsilon')
    parser.add_argument(
        '--pgd-video',
        type=float,
        default=None,
        metavar='PV',
        help='whether to to produce a video of the agent performing under PGD attack with given epsilon')
    parser.add_argument('--video',
                        dest='video',
                        action='store_true',
                        help = 'saves a video of standard eval run of model')
    parser.add_argument('--fgsm',
                        dest='fgsm',
                        action='store_true',
                        help = 'evaluate against fast gradient sign attack')
    parser.add_argument('--target_fgsm',
                        dest='target_fgsm',
                        action='store_true',
                        help = 'evaluate against fast gradient sign targeted attack')
    parser.add_argument('--pgd',
                       dest='pgd',
                       action='store_true',
                       help='evaluate against projected gradient descent attack')
    parser.add_argument('--target_pgd',
                       dest='target_pgd',
                       action='store_true',
                       help='evaluate against projected gradient descent targeted attack')
    parser.add_argument('--gwc',
                       dest='gwc',
                       action='store_true',
                       help='whether to evaluate worst possible(greedy) outcome under any epsilon bounded attack')
    parser.add_argument('--action-pert',
                       dest='action_pert',
                       action='store_true',
                       help='whether to evaluate performance under action perturbations')
    parser.add_argument('--acr',
                       dest='acr',
                       action='store_true',
                       help='whether to evaluate the action certification rate of an agent')
    parser.add_argument('--nominal',
                       dest='nominal',
                       action='store_true',
                       help='evaluate the agents nominal performance without any adversaries')
    parser.add_argument('--target_attack', default=False, type=bool)
    parser.add_argument('--target_path', default=None, type=str)

    parser.add_argument('--attack_method', default='none', type=str,
                        help='none, random,'
                             'untarget_q_fgsm, untarget_q_pgd, target_q_fgsm, target_q_pgd,'
                             'untarget_kl_fgsm, untarget_kl_pgd, target_kl_fgsm, target_kl_pgd')
    parser.add_argument('--steps', default=10, type=int,
                        help='attack steps for pgd')

    parser.add_argument('--run', default=0, type=int)


    parser.set_defaults(video=False, fgsm=False, pgd=False, gwc=False, action_pert=False, acr=False)
    return parser.parse_args()


def run_attack(input, model, attack_method, epsilon, steps, target_model):
    if epsilon == 0.0:
        return input
    # print(attack_method)
    if attack_method == 'none':
        return input
    elif attack_method == 'random':
        noise = torch.empty_like(input).uniform_(epsilon, epsilon)
        return input + noise
    elif attack_method == 'untarget_q_fgsm':
        model.zero_grad()
        input.requires_grad = True
        _, output = model.forward(input)
        action = torch.argmax(output, dim=1)
        original_label = action[0].item()
        # print(output)
        # print(original_label)
        loss = output[0][original_label]
        loss.backward()
        # l_infty
        # input = input - torch.sign(input.grad) * epsilon
        # l_2
        input = input - input.grad / torch.norm(input.grad) * epsilon * 80.0
        return input
    elif attack_method == 'target_q_fgsm':
        target_model.zero_grad()
        _, output = target_model.forward(input)
        action = torch.argmax(output, dim=1)
        target_label = action[0].item()

        model.zero_grad()
        input.requires_grad = True
        _, output = model.forward(input)

        # print(output)
        # print(original_label)
        loss = output[0][target_label]
        loss.backward()
        # print(torch.sign(input.grad) * epsilon)
        # l_infty
        # input = input + torch.sign(input.grad) * epsilon
        # l_2
        input = input + input.grad / torch.norm(input.grad) * epsilon * 80.0
        return input
    elif attack_method == 'untarget_q_pgd':
        clamp_min = input - epsilon
        clamp_max = input + epsilon

        _, output = model.forward(input)
        action = torch.argmax(output, dim=1)
        original_label = action[0].item()

        epsilon_step = epsilon / steps
        input.requires_grad = True
        for _ in range(steps):
            input = input.clone().detach().requires_grad_()
            model.zero_grad()
            _, output = model.forward(input)
            loss = output[0][original_label]
            loss.backward()
            # print(torch.sign(input.grad) * epsilon)
            # l_infty
            # input = input - torch.sign(input.grad) * epsilon_step
            # l_2
            input = input - input.grad / torch.norm(input.grad) * epsilon_step * 80.0

            input = torch.min(torch.max(input, clamp_min), clamp_max)
            # print(input)
        return input
    elif attack_method == 'target_q_pgd':
        clamp_min = input - epsilon
        clamp_max = input + epsilon

        target_model.zero_grad()
        _, output = target_model.forward(input)
        action = torch.argmax(output, dim=1)
        target_label = action[0].item()

        model.zero_grad()
        epsilon_step = epsilon / steps
        # input.requires_grad = True
        for _ in range(steps):
            input = input.clone().detach().requires_grad_()
            model.zero_grad()
            _, output = model.forward(input)
            loss = output[0][target_label]
            loss.backward()
            # print(torch.sign(input.grad) * epsilon)

            # l_infty
            # input = input + torch.sign(input.grad) * epsilon_step
            # l_2
            input = input + input.grad / torch.norm(input.grad) * epsilon_step * 80.0

            input = torch.min(torch.max(input, clamp_min), clamp_max)
            # print(input)
        return input
    elif attack_method == 'untarget_kl_fgsm':
        with torch.no_grad():
            # input.requires_grad = True
            _, original_output = model.forward(input)
            # original_output = original_output[0]
            original_output = F.softmax(original_output, dim=1)
            # input.grad.data.zero()

        model.zero_grad()
        input = input.clone().detach().requires_grad_()
        _, output = model.forward(input)
        output = F.log_softmax(output, dim=1)
        # output = output[0]

        loss = F.kl_div(output, original_output, reduction='sum')
        # print(output)
        # print(original_output)
        # loss = ce(output, original_output)
        loss.backward()
        # l_infty
        # input = input + torch.sign(input.grad) * epsilon
        # l_2
        input = input + input.grad / torch.norm(input.grad) * epsilon * 80.0
        return input
    elif attack_method == 'target_kl_fgsm':
        with torch.no_grad():
            target_model.zero_grad()
            _, target_output = target_model.forward(input)
            target_output = F.softmax(target_output, dim=1)

        model.zero_grad()
        input = input.clone().detach().requires_grad_()
        _, output = model.forward(input)
        # output = F.log_softmax(output, dim=1)
        output = F.softmax(output, dim=1)
        # output = output[0]

        loss = -(output * torch.log(target_output)).sum()
        # loss = F.kl_div(output, target_output, reduction='sum')
        loss.backward()
        # l_infty
        # input = input - torch.sign(input.grad) * epsilon
        # l_2
        input = input - input.grad / torch.norm(input.grad) * epsilon * 80.0

        return input
    elif attack_method == 'untarget_kl_pgd':
        with torch.no_grad():
            clamp_min = input - epsilon
            clamp_max = input + epsilon

            # input.requires_grad = True
            _, original_output = model.forward(input)
            # original_output = original_output[0]
            original_output = F.softmax(original_output, dim=1)
            # input.grad.data.zero()

            epsilon_step = epsilon / steps

        for _ in range(steps):
            model.zero_grad()
            input = input.clone().detach().requires_grad_()
            _, output = model.forward(input)
            # output = F.log_softmax(output, dim=1)
            output = F.log_softmax(output, dim=1)

            loss = F.kl_div(output, original_output, reduction='sum')
            loss.backward()
            # l_infty
            # input = input + torch.sign(input.grad) * epsilon_step
            # l_2
            input = input + input.grad / torch.norm(input.grad) * epsilon_step * 80.0

            input = torch.min(torch.max(input, clamp_min), clamp_max)
            # print(input)
        return input
    elif attack_method == 'target_kl_pgd':
        with torch.no_grad():
            clamp_min = input - epsilon
            clamp_max = input + epsilon

            target_model.zero_grad()
            _, target_output = target_model.forward(input)
            target_output = F.softmax(target_output, dim=1)

            epsilon_step = epsilon / steps

        for _ in range(steps):
            model.zero_grad()
            input = input.clone().detach().requires_grad_()
            _, output = model.forward(input)
            # output = F.log_softmax(output, dim=1)
            output = F.softmax(output, dim=1)

            loss = -(output * torch.log(target_output)).sum()
            # loss = F.kl_div(output, target_output, reduction='sum')
            loss.backward()

            # l_infty
            # input = input - torch.sign(input.grad) * epsilon_step
            # l_2
            input = input - input.grad / torch.norm(input.grad) * epsilon_step * 80.0

            input = torch.min(torch.max(input, clamp_min), clamp_max)
        return input
    elif attack_method == 'untarget_pi_fgsm':
        model.zero_grad()
        input.requires_grad = True
        _, output = model.forward(input)
        action = torch.argmax(output, dim=1)
        original_label = action[0].item()
        output = F.softmax(output, dim=1)
        # print(output)
        # print(original_label)
        loss = output[0][original_label]
        loss.backward()
        # print(torch.sign(input.grad) * epsilon)
        # l_infty
        # input = input - torch.sign(input.grad) * epsilon
        # l_2
        input = input - input.grad / torch.norm(input.grad) * epsilon * 80.0

        return input
    elif attack_method == 'target_pi_fgsm':
        target_model.zero_grad()
        _, output = target_model.forward(input)
        action = torch.argmax(output, dim=1)
        target_label = action[0].item()

        model.zero_grad()
        input.requires_grad = True
        _, output = model.forward(input)
        output = F.softmax(output, dim=1)

        # print(output)
        # print(original_label)
        loss = output[0][target_label]
        loss.backward()
        # print(torch.sign(input.grad) * epsilon)

        # l_infty
        # input = input + torch.sign(input.grad) * epsilon
        # l_2
        input = input + input.grad / torch.norm(input.grad) * epsilon * 80.0

        return input
    elif attack_method == 'untarget_pi_pgd':
        clamp_min = input - epsilon
        clamp_max = input + epsilon

        _, output = model.forward(input)
        action = torch.argmax(output, dim=1)
        original_label = action[0].item()

        epsilon_step = epsilon / steps
        input.requires_grad = True
        for _ in range(steps):
            input = input.clone().detach().requires_grad_()
            model.zero_grad()
            _, output = model.forward(input)
            output = F.softmax(output, dim=1)
            loss = output[0][original_label]
            loss.backward()
            # print(torch.sign(input.grad) * epsilon)

            # l_infty
            # input = input - torch.sign(input.grad) * epsilon_step
            # l_2
            input = input - input.grad / torch.norm(input.grad) * epsilon_step * 80.0

            input = torch.min(torch.max(input, clamp_min), clamp_max)
            # print(input)
        return input
    elif attack_method == 'target_pi_pgd':
        clamp_min = input - epsilon
        clamp_max = input + epsilon

        target_model.zero_grad()
        _, output = target_model.forward(input)
        action = torch.argmax(output, dim=1)
        target_label = action[0].item()

        model.zero_grad()
        epsilon_step = epsilon / steps
        # input.requires_grad = True
        for _ in range(steps):
            input = input.clone().detach().requires_grad_()
            model.zero_grad()
            _, output = model.forward(input)
            output = F.softmax(output, dim=1)
            loss = output[0][target_label]
            loss.backward()
            # print(torch.sign(input.grad) * epsilon)
            # l_infty
            # input = input + torch.sign(input.grad) * epsilon_step
            # l_2
            input = input + input.grad / torch.norm(input.grad) * epsilon_step * 80.0

            input = torch.min(torch.max(input, clamp_min), clamp_max)
            # print(input)
        return input
    return input

            
def run_trajectory(model, env, attack_method, args, epsilon, target_model=None):
    assert attack_method in ('none', 'random', 'untarget_q_fgsm', 'untarget_q_pgd',
                             'target_q_fgsm', 'target_q_pgd', 'untarget_kl_fgsm',
                             'untarget_kl_pgd', 'target_kl_fgsm', 'target_kl_pgd',
                             'untarget_pi_fgsm', 'untarget_pi_pgd', 'target_pi_fgsm',
                             'target_pi_pgd'), 'Invalid attack type'
    loss_func = torch.nn.CrossEntropyLoss()
    m = PytorchModel(model, loss_func, (0, 1), channel_axis=1, nb_classes=env.action_space, device=args.gpu_id)

    total_count = 0
    episode_reward = 0
    state = env.reset()

    while True:
        total_count += 1
        if total_count % 1000 == 0 and total_count > 0:
            print('Now is step', total_count)
        input_x = torch.FloatTensor(state).unsqueeze(0)
        if args.gpu_id >= 0:
            with torch.cuda.device(args.gpu_id):
                input_x = input_x.cuda()

        input_x = run_attack(input_x, model, attack_method, epsilon, args.steps, target_model)
        _, output = model.forward(input_x)
        # print(output)

        action = torch.argmax(output, dim=1)
        next_state, reward, done, info = env.step(action[0])

        episode_reward += reward
        state = next_state
        if done and not info:
            state = env.reset()
        elif done:
            # print(info)
            print('Reward under {} attack {}'.format(attack_method, episode_reward))
            return episode_reward


def main(args):
    setup_json = read_config(args.env_config)
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env:
            env_conf = setup_json[i]
    task = args.env
    env = atari_env(args.env, env_conf, args)
    model = A3Cff(env.observation_space.shape[0], env.action_space)
    target_model = A3Cff(env.observation_space.shape[0], env.action_space)

    if args.gpu_id >= 0:
        weights = torch.load(args.load_path, map_location=torch.device('cuda:{}'.format(args.gpu_id)))
        model.load_state_dict(weights)
        with torch.cuda.device(args.gpu_id):
            model.cuda()
        if args.target_attack:
            target_weights = torch.load(args.target_path, map_location=torch.device('cuda:{}'.format(args.gpu_id)))
            target_model.load_state_dict(target_weights)
            with torch.cuda.device(args.gpu_id):
                target_model.cuda()
    else:
        weights = torch.load(args.load_path, map_location=torch.device('cpu'))
        model.load_state_dict(weights)
        if args.target_attack:
            target_weights = torch.load(args.target_path, map_location=torch.device('cpu'))
            target_model.load_state_dict(target_weights)
    model.eval()
    if args.target_attack:
        target_model.eval()
        neg_name = (args.target_path.split('/')[-1]).split('.')[0]

    save_name = (args.load_path.split('/')[-1]).split('.')[0]
    if not os.path.exists('videos'):
        os.mkdir('videos')
    if not os.path.exists('figures'):
        os.mkdir('figures')
    if not os.path.exists('figures/' + save_name):
        os.mkdir('figures/' + save_name)

    epsilons = [0.3 / 255, 1 / 255, 3 / 255, 8 / 255]
    if args.env == 'PongNoFrameskip-v4':
        epsilons = [0.0, 0.3 / 255, 0.5 / 255, 1 / 255, 2 / 255, 3 / 255, 4 / 255, 5 / 255, 8 / 255]
    elif args.env == 'QbertNoFrameskip-v4':
        # epsilons = [0.0, 0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
        epsilons = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
    elif args.env == 'BeamRiderNoFrameskip-v4':
        # epsilons = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
        epsilons = [0.0, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014]
    elif args.env == 'SpaceInvadersNoFrameskip-v4':
        # epsilons = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
        epsilons = [0.0, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014]
    print('env is', args.env)
    print('epsilons are', epsilons)
    print('attack method is', args.attack_method)
    print('save name', save_name)

    if args.target_attack:
        print('neg name:', neg_name)
        if not os.path.exists('data/' + task + '/' + args.attack_method):
            os.mkdir('data/' + task + '/' + args.attack_method)
        f = open('data/' + task + '/' + args.attack_method + '/' + save_name + '_' + neg_name + '.txt', "w")
    else:
        if not os.path.exists('data/' + task + '/' + args.attack_method):
            os.mkdir('data/' + task + '/' + args.attack_method)
        f = open('data/' + task + '/' + args.attack_method + '/' + save_name + '.txt', "w")

    print(args.attack_method, file=f)
    for i in epsilons:
        print(i, end=" ", file=f)
    print('', file=f)

    rewards = []
    for epsilon in epsilons:
        curr_rewards = []
        print('epsilon is', epsilon)
        for i in range(5):
            start_time = time.time()
            reward = run_trajectory(model, env, args.attack_method, args,
                                    epsilon, target_model)
            curr_rewards.append(reward)
            print('time cost:', time.time() - start_time)
        rewards.append(curr_rewards)
    rewards = np.array(rewards)
    mean_rewards = np.mean(rewards, axis=1)

    for i in mean_rewards:
        print(i, end=" ", file=f)
    print('', file=f)

    std_rewards = np.std(rewards, axis=1)
    for i in std_rewards:
        print(i, end=" ", file=f)
    print('', file=f)


if __name__ == '__main__':
    args = get_parser()
    attack1 = ['none', 'random', 'untarget_q_fgsm', 'untarget_q_pgd',
               'untarget_kl_fgsm', 'untarget_kl_pgd', 'untarget_pi_fgsm',
               'untarget_pi_pgd']
    attack2 = ['target_q_fgsm', 'target_q_pgd', 'target_kl_fgsm',
               'target_kl_pgd', 'target_pi_fgsm', 'target_pi_pgd']
    
    if args.env == 'PongNoFrameskip-v4':
        ta = "Pong"
    elif args.env == 'QbertNoFrameskip-v4':
        ta = "Qbert"
    elif args.env == 'BeamRiderNoFrameskip-v4':
        ta = "BeamRider"
    elif args.env == 'SpaceInvadersNoFrameskip-v4':
        ta = "SpaceInvaders"

    for method in attack1:
        args.attack_method = method
        for i in range(1):
            args.load_path = "trained_models/"+ta+"_best"+str(i+1)+".pt"
            main(args)

    for method in attack2:
        args.attack_method = method
        for i in range(1):
            args.load_path = "trained_models/"+ta+"_best" + str(i + 1) + ".pt"
            args.target_attack = True
            for j in range(2):
                args.target_path = "negpolicy/"+ta+"_neg_best"+str(j+4)+".pt"
                main(args)
