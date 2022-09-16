python try.py --config-path configs/config_ant_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-ant.model --attack-method none --deterministic --attack-advpolicy-network models/atla_release/PPO/attack-ppo-ant.model --attack_methods "none critic random action advpolicy target_action target_action_advpolicy" --task_algorithm 'ant_vanilla_ppo'
python try.py --config-path configs/config_ant_atla_ppo.json --load-model models/atla_release/ATLA-PPO/model-atla-ppo-ant.model --attack-method none --deterministic --attack-advpolicy-network models/atla_release/ATLA-PPO/attack-atla-ppo-ant.model --attack_methods "none critic random action advpolicy target_action target_action_advpolicy" --task_algorithm 'ant_atla_ppo'
python try.py --config-path configs/config_halfcheetah_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-halfcheetah.model --attack-method none --deterministic --attack-advpolicy-network models/atla_release/PPO/attack-ppo-halfcheetah.model --attack_methods "none critic random action advpolicy target_action target_action_advpolicy" --task_algorithm 'halfcheetah_vanilla_ppo'
python try.py --config-path configs/config_halfcheetah_atla_ppo.json --load-model models/atla_release/ATLA-PPO/model-atla-ppo-halfcheetah.model --attack-method none --deterministic --attack-advpolicy-network models/atla_release/ATLA-PPO/attack-atla-ppo-halfcheetah.model --attack_methods "none critic random action advpolicy target_action target_action_advpolicy" --task_algorithm 'halfcheetah_atla_ppo'
python try.py --config-path configs/config_hopper_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-hopper.model --attack-method none --deterministic --attack-advpolicy-network models/atla_release/PPO/attack-ppo-hopper.model --attack_methods "none critic random action advpolicy target_action target_action_advpolicy" --task_algorithm 'hopper_vanilla_ppo'
python try.py --config-path configs/config_hopper_atla_ppo.json --load-model models/atla_release/ATLA-PPO/model-atla-ppo-hopper.model --attack-method none --deterministic --attack-advpolicy-network models/atla_release/ATLA-PPO/attack-atla-ppo-hopper.model --attack_methods "none critic random action advpolicy target_action target_action_advpolicy" --task_algorithm 'hopper_atla_ppo'
python try.py --config-path configs/config_walker_vanilla_ppo.json --load-model models/atla_release/PPO/model-ppo-walker.model --attack-method none --deterministic --attack-advpolicy-network models/atla_release/PPO/attack-ppo-walker.model --attack_methods "none critic random action advpolicy target_action target_action_advpolicy" --task_algorithm 'walker_vanilla_ppo'
python try.py --config-path configs/config_walker_atla_ppo.json --load-model models/atla_release/ATLA-PPO/model-atla-ppo-walker.model --attack-method none --deterministic --attack-advpolicy-network models/atla_release/ATLA-PPO/attack-atla-ppo-walker.model --attack_methods "none critic random action advpolicy target_action target_action_advpolicy" --task_algorithm 'walker_atla_ppo'