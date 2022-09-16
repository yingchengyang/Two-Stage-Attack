import free_mjc
import free_mjc
import pickle
import sqlite3
from policy_gradients.agent import Trainer
import git
import numpy as np
import os
import copy
import random
import argparse
from policy_gradients import models
from policy_gradients.torch_utils import ZFilter
import sys
import json
import torch
import torch.optim as optim
from cox.store import Store, schema_from_dict
from run import main, add_common_parser_opts, override_json_params
from auto_LiRPA.eps_scheduler import LinearScheduler
import logging

from test import get_parser
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# colors for plot
COLORS = ([
    # deepmind style
    '#0072B2',
    '#009E73',
    '#D55E00',
    '#CC79A7',
    '#F0E442',
    # built-in color
    'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
    'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender', 'turquoise',
    'darkgreen', 'tan', 'salmon', 'gold',  'darkred', 'darkblue',
    # personal color
    '#313695',  # DARK BLUE
    '#74add1',  # LIGHT BLUE
    '#4daf4a',  # GREEN
    '#f46d43',  # ORANGE
    '#d73027',  # RED
    '#984ea3',  # PURPLE
    '#f781bf',  # PINK
    '#ffc832',  # YELLOW
    '#000000',  # BLACK
])


def main(params):
    override_params = copy.deepcopy(params)
    excluded_params = ['config_path', 'out_dir_prefix', 'num_episodes', 'row_id', 'exp_id',
                       'load_model', 'seed', 'deterministic', 'noise_factor', 'compute_kl_cert',
                       'use_full_backward', 'sqlite_path', 'early_terminate',
                       'negative', 'negative_id', 'attack_methods', 'task_algorithm']
    sarsa_params = ['sarsa_enable', 'sarsa_steps', 'sarsa_eps', 'sarsa_reg', 'sarsa_model_path']
    imit_params = ['imit_enable', 'imit_epochs', 'imit_model_path', 'imit_lr']

    # original_params contains all flags in config files that are overridden via command.
    for k in list(override_params.keys()):
        if k in excluded_params:
            del override_params[k]

    if params['sqlite_path']:
        print(f"Will save results in sqlite database in {params['sqlite_path']}")
        connection = sqlite3.connect(params['sqlite_path'])
        cur = connection.cursor()
        cur.execute('''create table if not exists attack_results
              (method varchar(20),
              mean_reward real,
              std_reward real,
              min_reward real,
              max_reward real,
              sarsa_eps real,
              sarsa_reg real,
              sarsa_steps integer,
              deterministic bool,
              early_terminate bool)''')
        connection.commit()
        # We will set this flag to True we break early.
        early_terminate = False

    # Append a prefix for output path.
    if params['out_dir_prefix']:
        params['out_dir'] = os.path.join(params['out_dir_prefix'], params['out_dir'])
        print(f"setting output dir to {params['out_dir']}")

    if params['config_path']:
        # Load from a pretrained model using existing config.
        # First we need to create the model using the given config file.
        json_params = json.load(open(params['config_path']))

        params = override_json_params(params, json_params, excluded_params + sarsa_params + imit_params)

    if params['sarsa_enable']:
        assert params['attack_method'] == "none" or params['attack_method'] is None, \
            "--train-sarsa is only available when --attack-method=none, but got {}".format(params['attack_method'])

    if 'load_model' in params and params['load_model']:
        for k, v in zip(params.keys(), params.values()):
            assert v is not None, f"Value for {k} is None"

        # Create the agent from config file.
        p = Trainer.agent_from_params(params, store=None)
        print('Loading pretrained model', params['load_model'])
        pretrained_model = torch.load(params['load_model'])
        if 'policy_model' in pretrained_model:
            p.policy_model.load_state_dict(pretrained_model['policy_model'])
        if 'val_model' in pretrained_model:
            p.val_model.load_state_dict(pretrained_model['val_model'])
        if 'policy_opt' in pretrained_model:
            p.POLICY_ADAM.load_state_dict(pretrained_model['policy_opt'])
        if 'val_opt' in pretrained_model:
            p.val_opt.load_state_dict(pretrained_model['val_opt'])
        # Restore environment parameters, like mean and std.
        if 'envs' in pretrained_model:
            p.envs = pretrained_model['envs']
        for e in p.envs:
            e.normalizer_read_only = True
            e.setup_visualization(params['show_env'], params['save_frames'], params['save_frames_path'])
    else:
        # Load from experiment directory. No need to use a config.
        base_directory = params['out_dir']
        store = Store(base_directory, params['exp_id'], mode='r')
        if params['row_id'] < 0:
            row = store['final_results'].df
        else:
            checkpoints = store['checkpoints'].df
            row_id = params['row_id']
            row = checkpoints.iloc[row_id:row_id + 1]
        print("row to test: ", row)
        if params['cpu'] == None:
            cpu = False
        else:
            cpu = params['cpu']
        p, _ = Trainer.agent_from_data(store, row, cpu, extra_params=params, override_params=override_params,
                                       excluded_params=excluded_params)
        store.close()

    negative_policies = []
    if params['negative']:
        # Load from experiment directory. No need to use a config.
        base_directory = params['out_dir']
        negative_ids = params['negative_id'].split(' ')
        for negative_id in negative_ids:
            store = Store(base_directory, negative_id, mode='r')
            if params['row_id'] < 0:
                row = store['final_results'].df
            else:
                checkpoints = store['checkpoints'].df
                row_id = params['row_id']
                row = checkpoints.iloc[row_id:row_id + 1]
            print("row to test: ", row)
            if params['cpu'] == None:
                cpu = False
            else:
                cpu = params['cpu']
            neg_p, _ = Trainer.agent_from_data(store, row, cpu, extra_params=params, override_params=override_params,
                                           excluded_params=excluded_params)
            # p.negative_policy.append(neg_p.policy_model)
            negative_policies.append(neg_p.policy_model)
            store.close()
        if params['attack_method'] == "target_action_negpolicies":
            p.negative_policies = negative_policies

    rewards = []

    print('Gaussian noise in policy:')
    print(torch.exp(p.policy_model.log_stdev))
    original_stdev = p.policy_model.log_stdev.clone().detach()
    if params['noise_factor'] != 1.0:
        p.policy_model.log_stdev.data[:] += np.log(params['noise_factor'])
    if params['deterministic']:
        print('Policy runs in deterministic mode. Ignoring Gaussian noise.')
        p.policy_model.log_stdev.data[:] = -100
    print('Gaussian noise in policy (after adjustment):')
    print(torch.exp(p.policy_model.log_stdev))


    num_episodes = params['num_episodes']
    all_mean_reward = []
    all_std_reward = []
    all_kls = []
    for j in range(len(negative_policies)):
        p.kls = []
        all_rewards = []
        all_lens = []
        p.negative_policy = negative_policies[j]
        for i in range(num_episodes):
            print('Negative policy %d, Episode %d / %d' % (j + 1, i + 1, num_episodes))
            ep_length, ep_reward, actions, action_means, states, kl_certificates = p.run_test(
                compute_bounds=params['compute_kl_cert'], use_full_backward=params['use_full_backward'],
                original_stdev=original_stdev)
            if i == 0:
                all_actions = actions.copy()
                all_states = states.copy()
            else:
                all_actions = np.concatenate((all_actions, actions), axis=0)
                all_states = np.concatenate((all_states, states), axis=0)
            all_rewards.append(ep_reward)
            all_lens.append(ep_length)
            # Current step mean, std, min and max
            mean_reward, std_reward, min_reward, max_reward = np.mean(all_rewards), np.std(all_rewards), np.min(
                    all_rewards), np.max(all_rewards)


        print(params)

        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        kl_mean = np.mean(p.kls)

        print('\n')
        print('all rewards:', all_rewards)
        all_mean_reward.append(mean_reward)
        all_std_reward.append(std_reward)
        all_kls.append(kl_mean)


        print('all mean reward:', all_mean_reward)
        print('all std reward:', all_std_reward)
        aaa = []
        for i in range(len(all_mean_reward)):
            aaa.append((all_mean_reward[i], all_std_reward[i]))
        aaa = np.array(aaa)
        print(aaa)
        b = aaa[:, 0]
        index = np.lexsort((b,))
        print(aaa[index])
        mean_reward = aaa[index][2][0]
        std_reward = aaa[index][2][1]
        print('rewards stats:\nmean: {}, std:{}'.format(mean_reward, std_reward))

    return mean_reward, std_reward



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.load_model:
        assert args.config_path, "Need to specificy a config file when loading a pretrained model."

    if args.early_terminate:
        assert args.sqlite_path != '', "Need to specify --sqlite-path to terminate early."

    if args.sarsa_enable:
        if args.sqlite_path != '':
            print("When --sarsa-enable is specified, --sqlite-path and --early-terminate will be ignored.")

    params = vars(args)
    seed = params['seed']
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    attack_method = 'target_action_negpolicy'
    params['attack_method'] = attack_method
    params['attack_eps'] = 0.15
    print("attack method is", attack_method)
    print("attack eps is", params["attack_eps"])
    mean_reward, std_reward = main(params)

    print(mean_reward)
    print(std_reward)
