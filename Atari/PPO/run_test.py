from adversarial_test import run_main, parser


if __name__ == '__main__':
    args = parser.parse_args()
    if args.env_name == 'PongNoFrameskip-v4':
        epsilons = [0.0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014]
    elif args.env_name == 'QbertNoFrameskip-v4':
        epsilons = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
    elif args.env_name == 'BeamRiderNoFrameskip-v4':
        epsilons = [0.0, 0.012, 0.014, 0.016, 0.018, 0.02, 0.022]
    elif args.env_name == 'SpaceInvadersNoFrameskip-v4':
        epsilons = [0.0, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018]

    attack1 = ['none', 'random', 'untarget_kl_fgsm', 'untarget_kl_pgd',
               'untarget_pi_fgsm', 'untarget_pi_pgd']
    attack2 = ['target_kl_fgsm', 'target_kl_pgd']
    if args.env_name == 'PongNoFrameskip-v4':
        ta = "Pong"
    elif args.env_name == 'QbertNoFrameskip-v4':
        ta = "Qbert"
    elif args.env_name == 'BeamRiderNoFrameskip-v4':
        ta = "BeamRider"
    elif args.env_name == 'SpaceInvadersNoFrameskip-v4':
        ta = "SpaceInvaders"

    # run all types of attack
    if args.run == 0:
        for method in attack1:
            args.attack_method = method
            for i in range(5):
                args.load_dir = "trained_models/ppo/"+ta+"_"+str(i+1)+".pt"
                run_main(args, epsilons)

        for method in attack2:
            args.attack_method = method
            for i in range(5):
                args.load_dir = "trained_models/ppo/"+ta+"_"+str(i+1)+".pt"
                args.target_attack = True
                for j in range(5):
                    args.target_path = "trained_models/ppo_neg_policy/"+ta+"_neg_"+str(j+1)+".pt"
                    run_main(args, epsilons)

    # only run baselines
    elif args.run == 1:
        for method in attack1:
            args.attack_method = method
            for i in range(5):
                args.load_dir = "trained_models/ppo/"+ta+"_"+str(i+1)+".pt"
                run_main(args, epsilons)

    # only run Two-Stage attack
    elif args.run == 2:
        for method in attack2:
            args.attack_method = method
            for i in range(5):
                args.load_dir = "trained_models/ppo/"+ta+"_"+str(i+1)+".pt"
                args.target_attack = True
                for j in range(5):
                    args.target_path = "trained_models/ppo_neg_policy/" + ta + "_neg_" + str(j + 1) + ".pt"
                    run_main(args, epsilons)