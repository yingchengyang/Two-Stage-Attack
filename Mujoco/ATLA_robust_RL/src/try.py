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

    if params['sarsa_enable']:
        num_steps = params['sarsa_steps']
        # learning rate scheduler: linearly annealing learning rate after
        lr_decrease_point = num_steps * 2 / 3
        decreasing_steps = num_steps - lr_decrease_point
        lr_sch = lambda epoch: 1.0 if epoch < lr_decrease_point else (
                                                                                 decreasing_steps - epoch + lr_decrease_point) / decreasing_steps
        # robust training scheduler. Currently using 1/3 epochs for warmup, 1/3 for schedule and 1/3 for final training.
        eps_start_point = int(num_steps * 1 / 3)
        robust_eps_scheduler = LinearScheduler(params['sarsa_eps'], f"start={eps_start_point},length={eps_start_point}")
        robust_beta_scheduler = LinearScheduler(1.0, f"start={eps_start_point},length={eps_start_point}")
        # reinitialize value model, and run value function learning steps.
        p.setup_sarsa(lr_schedule=lr_sch, eps_scheduler=robust_eps_scheduler, beta_scheduler=robust_beta_scheduler)
        # Run Sarsa training.
        for i in range(num_steps):
            print(f'Step {i + 1} / {num_steps}, lr={p.sarsa_scheduler.get_last_lr()}')
            mean_reward = p.sarsa_step()
            rewards.append(mean_reward)
            # for w in p.val_model.parameters():
            #     print(f'{w.size()}, {torch.norm(w.view(-1), 2)}')
        # Save Sarsa model.
        saved_model = {
            'state_dict': p.sarsa_model.state_dict(),
            'metadata': params,
        }
        torch.save(saved_model, params['sarsa_model_path'])
    elif params['imit_enable']:
        num_epochs = params['imit_epochs']
        num_episodes = params['num_episodes']
        print('\n\n' + 'Start collecting data\n' + '-' * 80)
        for i in range(num_episodes):
            print('Collecting %d / %d episodes' % (i + 1, num_episodes))
            ep_length, ep_reward, actions, action_means, states, kl_certificates = p.run_test(
                compute_bounds=params['compute_kl_cert'], use_full_backward=params['use_full_backward'],
                original_stdev=original_stdev)
            not_dones = np.ones(len(actions))
            not_dones[-1] = 0
            if i == 0:
                all_actions = actions.copy()
                all_states = states.copy()
                all_not_dones = not_dones.copy()
            else:
                all_actions = np.concatenate((all_actions, actions), axis=0)
                all_states = np.concatenate((all_states, states), axis=0)
                all_not_dones = np.concatenate((all_not_dones, not_dones))
        print('Collected actions shape:', all_actions.shape)
        print('Collected states shape:', all_states.shape)
        p.setup_imit(lr=params['imit_lr'])
        p.imit_steps(torch.from_numpy(all_actions), torch.from_numpy(all_states), torch.from_numpy(all_not_dones),
                     num_epochs)
        saved_model = {
            'state_dict': p.imit_network.state_dict(),
            'metadata': params,
        }
        torch.save(saved_model, params['imit_model_path'])
    elif params['attack_method'] == "target_action_negpolicy":
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

                if i > num_episodes // 5 and params['early_terminate'] and params['sqlite_path'] and params[
                    'attack_method'] != 'none':
                    # Attempt to early terminiate if some other attacks have done with low reward.
                    cur.execute("SELECT MIN(mean_reward) FROM attack_results WHERE deterministic=?;",
                                (params['deterministic'],))
                    current_best_reward = cur.fetchone()[0]
                    print(f'current best: {current_best_reward}, ours: {mean_reward} +/- {std_reward}, min: {min_reward}')
                    # Terminiate if mean - 2*std is worse than best, or our min is worse than best.
                    if current_best_reward is not None and ((current_best_reward < mean_reward - 2 * std_reward) or
                                                            (min_reward > current_best_reward)):
                        print('terminating early!')
                        early_terminate = True
                        break

            attack_dir = 'attack-{}-eps-{}'.format(params['attack_method'], params['attack_eps'])
            if 'sarsa' in params['attack_method']:
                attack_dir += '-sarsa_steps-{}-sarsa_eps-{}-sarsa_reg-{}'.format(params['sarsa_steps'], params['sarsa_eps'],
                                                                                 params['sarsa_reg'])
                if 'action' in params['attack_method']:
                    attack_dir += '-attack_sarsa_action_ratio-{}'.format(params['attack_sarsa_action_ratio'])
            save_path = os.path.join(params['out_dir'], params['exp_id'], attack_dir)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for name, value in [('actions', all_actions), ('states', all_states), ('rewards', all_rewards),
                                ('length', all_lens)]:
                with open(os.path.join(save_path, '{}.pkl'.format(name)), 'wb') as f:
                    pickle.dump(value, f)
            print(params)
            with open(os.path.join(save_path, 'params.json'), 'w') as f:
                json.dump(params, f, indent=4)

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
        # mean_reward = all_mean_reward[0]
        # std_reward = all_std_reward[0]
        # kl = all_kls[0]
        # for j in range(len(all_mean_reward)):
        #     if all_kls[j] > kl:
        #         kl = all_kls[j]
        #         mean_reward = all_mean_reward[j]
        #         std_reward = all_std_reward[j]
        #     # if all_mean_reward[j] < mean_reward:
        #     #     mean_reward = all_mean_reward[j]
        #     #     std_reward = all_std_reward[j]
        # print('all mean rewards: ', all_mean_reward)
        # print('all std rewards: ', all_std_reward)
        # print('all kls: ', all_kls)
        # print('rewards stats:\nmean: {}, std:{}'.format(mean_reward, std_reward))
        return mean_reward, std_reward
    else:
        num_episodes = params['num_episodes']
        all_rewards = []
        all_lens = []
        all_kl_certificates = []

        for i in range(num_episodes):
            print('Episode %d / %d' % (i + 1, num_episodes))
            ep_length, ep_reward, actions, action_means, states, kl_certificates = p.run_test(
                compute_bounds=params['compute_kl_cert'], use_full_backward=params['use_full_backward'],
                original_stdev=original_stdev)
            if i == 0:
                all_actions = actions.copy()
                all_states = states.copy()
            else:
                all_actions = np.concatenate((all_actions, actions), axis=0)
                all_states = np.concatenate((all_states, states), axis=0)
            if params['compute_kl_cert']:
                print('Epoch KL certificates:', kl_certificates)
                all_kl_certificates.append(kl_certificates)
            all_rewards.append(ep_reward)
            all_lens.append(ep_length)
            # Current step mean, std, min and max
            mean_reward, std_reward, min_reward, max_reward = np.mean(all_rewards), np.std(all_rewards), np.min(
                all_rewards), np.max(all_rewards)

            if i > num_episodes // 5 and params['early_terminate'] and params['sqlite_path'] and params[
                'attack_method'] != 'none':
                # Attempt to early terminiate if some other attacks have done with low reward.
                cur.execute("SELECT MIN(mean_reward) FROM attack_results WHERE deterministic=?;",
                            (params['deterministic'],))
                current_best_reward = cur.fetchone()[0]
                print(f'current best: {current_best_reward}, ours: {mean_reward} +/- {std_reward}, min: {min_reward}')
                # Terminiate if mean - 2*std is worse than best, or our min is worse than best.
                if current_best_reward is not None and ((current_best_reward < mean_reward - 2 * std_reward) or
                                                        (min_reward > current_best_reward)):
                    print('terminating early!')
                    early_terminate = True
                    break

        attack_dir = 'attack-{}-eps-{}'.format(params['attack_method'], params['attack_eps'])
        if 'sarsa' in params['attack_method']:
            attack_dir += '-sarsa_steps-{}-sarsa_eps-{}-sarsa_reg-{}'.format(params['sarsa_steps'], params['sarsa_eps'],
                                                                             params['sarsa_reg'])
            if 'action' in params['attack_method']:
                attack_dir += '-attack_sarsa_action_ratio-{}'.format(params['attack_sarsa_action_ratio'])
        save_path = os.path.join(params['out_dir'], params['exp_id'], attack_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for name, value in [('actions', all_actions), ('states', all_states), ('rewards', all_rewards),
                            ('length', all_lens)]:
            with open(os.path.join(save_path, '{}.pkl'.format(name)), 'wb') as f:
                pickle.dump(value, f)
        print(params)
        with open(os.path.join(save_path, 'params.json'), 'w') as f:
            json.dump(params, f, indent=4)

        mean_reward, std_reward, min_reward, max_reward = np.mean(all_rewards), np.std(all_rewards), np.min(
            all_rewards), np.max(all_rewards)
        if params['compute_kl_cert']:
            print('KL certificates stats: mean: {}, std: {}, min: {}, max: {}'.format(np.mean(all_kl_certificates),
                                                                                      np.std(all_kl_certificates),
                                                                                      np.min(all_kl_certificates),
                                                                                      np.max(all_kl_certificates)))
        # write results to sqlite.
        if params['sqlite_path']:
            method = params['attack_method']
            if params['attack_method'] == "sarsa":
                # Load sarsa parameters from checkpoint
                sarsa_ckpt = torch.load(params['attack_sarsa_network'])
                sarsa_meta = sarsa_ckpt['metadata']
                sarsa_eps = sarsa_meta['sarsa_eps'] if 'sarsa_eps' in sarsa_meta else -1.0
                sarsa_reg = sarsa_meta['sarsa_reg'] if 'sarsa_reg' in sarsa_meta else -1.0
                sarsa_steps = sarsa_meta['sarsa_steps'] if 'sarsa_steps' in sarsa_meta else -1
            elif params['attack_method'] == "sarsa+action":
                sarsa_eps = -1.0
                sarsa_reg = params['attack_sarsa_action_ratio']
                sarsa_steps = -1
            else:
                sarsa_eps = -1.0
                sarsa_reg = -1.0
                sarsa_steps = -1
            try:
                cur.execute("INSERT INTO attack_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                            (method, mean_reward, std_reward, min_reward, max_reward, sarsa_eps, sarsa_reg, sarsa_steps,
                             params['deterministic'], early_terminate))
                connection.commit()
            except sqlite3.OperationalError as e:
                import traceback
                traceback.print_exc()
                print('Cannot insert into the SQLite table. Give up.')
            else:
                print(f'results saved to database {params["sqlite_path"]}')
            connection.close()
        print('\n')
        print('all rewards:', all_rewards)
        print(
            'rewards stats:\nmean: {}, std:{}, min:{}, max:{}'.format(mean_reward, std_reward, min_reward, max_reward))
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

    attack_eps = np.linspace(0.00, 0.15, 16)
    # attack_eps = np.linspace(0.05, 0.10, 2)
    # attack_methods = ["none", "critic", "random", "action",
    #                   "advpolicy", "target_action", "target_action_advpolicy",
    #                   "target_action_negpolicy"]
    # attack_methods = ["none", "critic", "random", "action",
    #                   "advpolicy", "target_action", "target_action_advpolicy"]
    # attack_methods = ["target_action_negpolicy"]
    # attack_methods = ["advpolicy", "target_action_advpolicy"]
    attack_methods = params['attack_methods'].split(' ')
    print('attack methods are ', attack_methods)

    plt.figure()
    # print(params['task_algorithm'] + ".txt")
    # print(str(params['task_algorithm']))
    # a = params['task_algorithm'] + '.txt'
    # print(a)
    f = open('./'+params['task_algorithm']+'.txt', "w")
    print(params['task_algorithm']+".txt")
    for i in attack_eps:
        print(i, end=" ", file=f)
    print('', file=f)
    for j in np.arange(len(attack_methods)):
        attack_method = attack_methods[j]
        params['attack_method'] = attack_method
        mean_rewards = []
        std_rewards = []
        for i in attack_eps:
            params['attack_eps'] = i
            print("attack method is", attack_method)
            print("attack eps is", params["attack_eps"])
            mean_reward, std_reward = main(params)
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)

        print(mean_rewards)
        print(std_rewards)
        mean_rewards = np.array(mean_rewards)
        std_rewards = np.array(std_rewards)
        plt.plot(attack_eps, mean_rewards, label=attack_method, color=COLORS[j])
        plt.fill_between(attack_eps, mean_rewards-std_rewards, mean_rewards+std_rewards,
                         color=COLORS[j], alpha=.4)
        print(attack_method, file=f)
        for i in mean_rewards:
            print(i, end=" ", file=f)
        print('', file=f)
        for i in std_rewards:
            print(i, end=" ", file=f)
        print('', file=f)
    plt.legend(loc=3, bbox_to_anchor=(1.05, 0), borderaxespad=0.)
    plt.savefig("./"+params['task_algorithm']+".png", bbox_inches='tight')
    f.close()