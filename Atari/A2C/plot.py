import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

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


def str_to_float(old_list):
    new_list = []
    for element in old_list:
        new_list.append(float(element))
    return new_list


if __name__ == '__main__':
    envs = ['BeamRider', 'Pong', 'Qbert', 'SpaceInvaders']
    fig = plt.figure(figsize=(24.8, 6.8))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.style.use('seaborn')
    plt.subplots_adjust(hspace=1.0)
    for _ in range(4):
        plt.subplot(1, 4, _ + 1)
        plt.tight_layout()
        env = envs[_]  # index = 8
        # env = 'Pong'  # index = 4
        # env = 'Qbert'  # index = 7
        # env = 'BeamRider' # index = 8
        index = 0
        task = env +'NoFrameskip-v4'
        print('task is', task)
        untarget_attack = ['none', 'random', 'untarget_q_fgsm', 'untarget_q_pgd',
                           'untarget_kl_fgsm', 'untarget_kl_pgd', 'untarget_pi_fgsm',
                           'untarget_pi_pgd']
        # target_attack = ['target_q_fgsm', 'target_q_pgd', 'target_kl_fgsm',
        #                 'target_kl_pgd', 'target_pi_fgsm', 'target_pi_pgd']
        target_attack = ['target_kl_fgsm', 'target_kl_pgd']
        # target_attack = []

        # epsilons = np.array(epsilons)
        if env == 'SpaceInvaders':
            plt.xlim((0, 0.018))
            plt.ylim((0, 800))
            plt.xticks(np.arange(0, 0.021, 0.003), size=16)
            plt.yticks(np.arange(0, 1000, 200), size=16)
        elif env == 'Pong':
            plt.xlim((0, 0.014))
            plt.ylim((-22, 22))
            plt.xticks(np.arange(0, 0.016, 0.002), size=16)
            plt.yticks(np.arange(-20, 25, 5), size=16)
        elif env == 'Qbert':
            plt.xlim((0, 0.007))
            plt.ylim((0, 16000))
            plt.xticks(np.arange(0, 0.008, 0.001), size=16)
            plt.yticks(np.arange(0, 17500, 2500), size=16)
        elif env == 'BeamRider':
            plt.xlim((0, 0.022))
            plt.ylim((0, 17500))
            plt.xticks(np.arange(0, 0.024, 0.004), size=16)
            plt.yticks(np.arange(0, 17500, 2500), size=16)

        means = []
        stds = []
        METHOD_NUM = 0

        # print('epsilon is:', epsilons[index])

        for method in untarget_attack:
            now_data = []
            now_result = []
            for i in range(5):
                f = open('data/' + task + '/' + method + '/' +env + '_best' + str(i+1) + '.txt', "r")
                for line in f:
                    now_data.append(line[:-1])
                f.close()
            for i in np.arange(0, len(now_data), 4):
                now_result.append(str_to_float(now_data[i+2].split()))
            epsilons = str_to_float(now_data[1].split())
            epsilons = np.array(epsilons)
            now_result = np.array(now_result)
            now_mean = now_result.mean(axis=0)
            now_std = now_result.std(axis=0)

            name = method
            if method == 'untarget_kl_fgsm':
                name = 'Stochastic MAD+FGSM'
            elif method == 'untarget_kl_pgd':
                name = 'Stochastic MAD+PGD'
            elif method == 'untarget_pi_fgsm':
                name = 'Deterministic MAD+FGSM'
            elif method == 'untarget_pi_pgd':
                name = 'Deterministic MAD+PGD'
            elif method == 'untarget_q_fgsm':
                name = 'Critic+FGSM'
            elif method == 'untarget_q_pgd':
                name = 'Critic+PGD'
            elif method == 'none':
                name = 'No Noise'
            elif method == 'random':
                name = 'Random'

            plt.plot(epsilons, now_mean, label=name, color=COLORS[METHOD_NUM], linewidth=3)
            plt.fill_between(epsilons, now_mean-now_std, now_mean+now_std,
                             color=COLORS[METHOD_NUM], alpha=.2)
            METHOD_NUM += 1

            print(method, 'mean:', now_mean[index], 'std:', now_std[index])

        for method in target_attack:
            now_data = []
            now_result = []
            for i in range(5):
                for j in range(5):
                    f = open('data/' + task + '/' + method + '/' +env + '_best' + str(i+1) +
                             '_' + env + '_neg_best' + str(j+1) +'.txt', "r")
                    for line in f:
                        now_data.append(line[:-1])
                    f.close()
            # print(now_data)
            for i in np.arange(0, len(now_data), 20):
                result = []
                for j in range(5):
                    result.append(str_to_float(now_data[i+2+j*4].split()))
                # print(result)
                result = np.array(result)
                result = np.sort(result, axis=0)
                # print(result)
                now_result.append(result[2])
            now_result = np.array(now_result)
            now_mean = now_result.mean(axis=0)
            now_std = now_result.std(axis=0)

            name = method
            if method == 'target_kl_fgsm':
                name = 'Our two-stage+FGSM'
            elif method == 'target_kl_pgd':
                name = 'Our two-stage+PGD'

            plt.plot(epsilons, now_mean, label=name, color=COLORS[METHOD_NUM], linewidth=3)
            plt.fill_between(epsilons, now_mean-now_std, now_mean+now_std,
                             color=COLORS[METHOD_NUM], alpha=.2)
            METHOD_NUM += 1

            print(method, 'mean:', now_mean[index], 'std:', now_std[index])

        print('epsilons:', epsilons)
        print('epsilon is:', epsilons[index])

        plt.title(env + ', A2C', fontsize=30)
        plt.grid(linestyle='dashed')
        plt.xlabel("$\epsilon$", fontsize=30)  # x轴上的名字
        plt.ylabel("Average Reward", fontsize=30)  # y轴上的名字

    # plt.legend(loc=3, bbox_to_anchor=(1.01, 0), borderaxespad=0., fontsize=16)
    plt.legend(['No Noise', 'Random', 'Critic+FGSM', 'Critic+PGD',
                'Stochastic MAD+FGSM', 'Stochastic MAD+PGD',
                'Deterministic MAD+FGSM', 'Deterministic MAD+PGD',
                'Our two-stage+FGSM', 'Our two-stage+PGD'], loc='center', bbox_to_anchor=[-1.5, 1.2],
               ncol=8, fancybox=True, columnspacing=0.5, handletextpad=0.2,
               borderpad=0.15, fontsize='xx-large', labelspacing=0.15, handlelength=3.0)
    plt.savefig('./result/a2c.pdf', bbox_inches='tight')

    envs = ['BeamRider', 'Pong', 'Qbert', 'SpaceInvaders']
    fig = plt.figure(figsize=(24.8, 6.8))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.style.use('seaborn')
    plt.subplots_adjust(hspace=1.0)
    # epsilons = np.array(epsilons)
    for _ in range(4):
        plt.subplot(1, 4, _ + 1)
        plt.tight_layout()
        env = envs[_]  # index = 8
        # env = 'Pong'  # index = 4
        # env = 'Qbert'  # index = 7
        # env = 'BeamRider' # index = 8
        index = 0
        task = env + 'NoFrameskip-v4'
        print('task is', task)
        untarget_attack = ['none', 'random', 'untarget_q_fgsm', 'untarget_q_pgd',
                           'untarget_kl_fgsm', 'untarget_kl_pgd', 'untarget_pi_fgsm',
                           'untarget_pi_pgd']
        # target_attack = ['target_q_fgsm', 'target_q_pgd', 'target_kl_fgsm',
        #                 'target_kl_pgd', 'target_pi_fgsm', 'target_pi_pgd']
        target_attack = ['target_kl_fgsm', 'target_kl_pgd']
        # target_attack = []

        if env == 'SpaceInvaders':
            plt.xlim((0, 0.018))
            plt.ylim((-10, 800))
            plt.xticks(np.arange(0, 0.021, 0.003), size=16)
            plt.yticks(np.arange(0, 1000, 200), size=16)
            plt.hlines(0, 0, 0.018, colors="black",
                       linestyles="dashed", label='Lowest Reward of the Task')
        elif env == 'Pong':
            plt.xlim((0, 0.014))
            plt.ylim((-22, 22))
            plt.xticks(np.arange(0, 0.016, 0.002), size=16)
            plt.yticks(np.arange(-20, 25, 5), size=16)
            plt.hlines(-21, 0, 0.014, colors="black",
                       linestyles="dashed", label='Lowest Reward of the Task')
        elif env == 'Qbert':
            plt.xlim((0.004, 0.007))
            plt.ylim((-50, 800))
            plt.xticks(np.arange(0.004, 0.0075, 0.0005), size=16)
            plt.yticks(np.arange(0, 800, 100), size=16)
            plt.hlines(0, 0.004, 0.007, colors="black",
                       linestyles="dashed", label='Lowest Reward of the Task')
        elif env == 'BeamRider':
            plt.xlim((0.01, 0.022))
            plt.ylim((-100, 4000))
            plt.xticks(np.arange(0.01, 0.024, 0.002), size=16)
            plt.yticks(np.arange(0, 5000, 1000), size=16)
            plt.hlines(0, 0.01, 0.022, colors="black",
                       linestyles="dashed", label='Lowest Reward of the Task')

        means = []
        stds = []
        METHOD_NUM = 0

        # print('epsilon is:', epsilons[index])
        all_means = []
        all_stds = []
        for method in untarget_attack:
            now_data = []
            now_result = []
            for i in range(5):
                f = open('data/' + task + '/' + method + '/' +env + '_best' + str(i+1) + '.txt', "r")
                for line in f:
                    now_data.append(line[:-1])
                f.close()
            for i in np.arange(0, len(now_data), 4):
                now_result.append(str_to_float(now_data[i+2].split()))
            epsilons = str_to_float(now_data[1].split())
            epsilons = np.array(epsilons)
            now_result = np.array(now_result)
            now_mean = now_result.mean(axis=0)
            now_std = now_result.std(axis=0)

            all_means.append(now_mean)
            all_stds.append(now_std)

        all_means = np.array(all_means)
        all_stds = np.array(all_stds)
        index = np.argmin(all_means, axis=0)
        print(all_means)
        print(all_stds)
        print(index)
        best_means = []
        best_stds = []
        for i in range(len(index)):
            best_means.append(all_means[index[i]][i])
            best_stds.append(all_stds[index[i]][i])
        print(best_means)
        print(best_stds)
        best_means = np.array(best_means)
        best_stds = np.array(best_stds)

        plt.plot(epsilons, best_means, label='Best Baseline', color=COLORS[METHOD_NUM], linewidth=3)
        plt.fill_between(epsilons, best_means-best_stds, best_means+best_stds,
                             color=COLORS[METHOD_NUM], alpha=.2)
        METHOD_NUM += 1


        for method in target_attack:
            now_data = []
            now_result = []
            for i in range(5):
                for j in range(5):
                    f = open('data/' + task + '/' + method + '/' +env + '_best' + str(i+1) +
                             '_' + env + '_neg_best' + str(j+1) +'.txt', "r")
                    for line in f:
                        now_data.append(line[:-1])
                    f.close()
            # print(now_data)
            for i in np.arange(0, len(now_data), 20):
                result = []
                for j in range(5):
                    result.append(str_to_float(now_data[i+2+j*4].split()))
                # print(result)
                result = np.array(result)
                result = np.sort(result, axis=0)
                # print(result)
                now_result.append(result[2])
            now_result = np.array(now_result)
            now_mean = now_result.mean(axis=0)
            now_std = now_result.std(axis=0)

            name = method
            if method == 'target_kl_fgsm':
                name = 'Our two-stage+FGSM'
            elif method == 'target_kl_pgd':
                name = 'Our two-stage+PGD'

            plt.plot(epsilons, now_mean, label=name, color=COLORS[METHOD_NUM], linewidth=3)
            plt.fill_between(epsilons, now_mean-now_std, now_mean+now_std,
                             color=COLORS[METHOD_NUM], alpha=.2)
            METHOD_NUM += 1

            print(method, 'mean:', now_mean[index], 'std:', now_std[index])

        print('epsilons:', epsilons)
        print('epsilon is:', epsilons[index])

        plt.title(env + ', A2C', fontsize=30)
        plt.grid(linestyle='dashed')
        plt.xlabel("$\epsilon$", fontsize=30)  # x轴上的名字
        plt.ylabel("Average Reward", fontsize=30)  # y轴上的名字

    # plt.legend(loc=3, bbox_to_anchor=(1.01, 0), borderaxespad=0., fontsize=16)
    plt.legend(['Best Baseline', 'Our two-stage+FGSM', 'Our two-stage+PGD',
                'Lowest Reward of the Task'], loc='center', bbox_to_anchor=[-1.4, 1.1],
               ncol=8, fancybox=True, columnspacing=0.5, handletextpad=0.2,
               borderpad=0.15, fontsize='xx-large', labelspacing=0.15, handlelength=3.0)
    plt.savefig('./result/a2c_best.pdf', bbox_inches='tight')