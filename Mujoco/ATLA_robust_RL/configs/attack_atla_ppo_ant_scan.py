import os
import json

import sys
from utils import dict_product, iwt, generate_configs

with open("../src/MuJoCo.json") as f:
    BASE_CONFIG = json.load(f)

PARAMS = {
    "game": ["Ant-v2"],
    "mode": ["adv_ppo"],
    "out_dir": ["attack_atla_ant/agents"],
    "norm_rewards": ["returns"],
    "initialization": ["orthogonal"],
    "anneal_lr": [True, False],
    "ppo_lr_adam": [0.0],  # this disables policy learning and we run attacks only.
    "adv_clip_eps": [0.2, 0.4],
    "adv_entropy_coeff": [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
    "adv_ppo_lr_adam": [3e-6, 1e-5, 3e-5],
    "adv_val_lr": [1e-6, 3e-6, 1e-5],
    "save_iters": [100],
    "train_steps": [2441],
    "robust_ppo_eps": [0.15],  # used for attack
    "load_model": ["models/atla_release/ATLA-PPO/model-atla-ppo-ant.model"],  # models for attack
    "value_clipping": [True],
}

generate_configs(BASE_CONFIG, PARAMS)
