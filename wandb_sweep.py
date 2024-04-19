#!/usr/bin/env python3
import sys; sys.path.append('..')
import wandb
from wandb_setting import config; config()
from wandb_setting import settings
import getpass
model_save_dir = f'/tmp/gcfp_{getpass.getuser()}'


# ------------------- arguments ---------------------
predmodel_dist = 'normal'
flow = 'node'        
phi_dim = 0             


dataset = 'sample' 

train_start_step, train_steps, eval_start_step, \
    eval_steps, test_start_step, test_steps = 0, 300, 300, 60, 360, 120
    
predmodel = 'HAR'
ret_hist_path = f'data/{dataset}.csv'


sweep_config = {
    'program': 'run.py',
    'project': 'gcfp',
    'name': f'{dataset}_{predmodel}_{flow}_{train_steps}_{eval_steps}',
    'method': 'grid',
    'metric': {
        'name': 'nll_x_mean',
        'goal': 'minimize',
    },
    'parameters': {
        'd_hidden': {
            'values': ['8', '4,4', '8,8', '4,4,4'],
        },
        'time_length': {
            'values': [0.25, 0.5, 1.0, 2.0],
        },
        'lr': {
            'values': [0.005],
        },
        'nonlinearity': {
            'values': ['swish'],
        },
        'optimizer': {
            'values': [
                'adam',
                ],
        },
    },
    'command': [x for x in [
        '${env}',
        'python3',
        '${program}',
        f'--ret_hist_path={ret_hist_path}',
        f'--train_start_step={train_start_step}',
        f'--train_steps={train_steps}',
        f'--eval_start_step={eval_start_step}',
        f'--eval_steps={eval_steps}',
        f'--test_start_step={test_start_step}',
        f'--test_steps={test_steps}',
        f'--n_iters=200',
        f'--predmodel={predmodel}',
        f'--predmodel_dist={predmodel_dist}',
        f'--flow={flow}',
        '--eval_interval=5',
        '--dim=1',
        '--sz_batch=9999',
        '--grad_clip=10.0',
        '--n_sampling_paths=32',
        '--momentcal_sampling_paths=32',
        f'--phi_dim={phi_dim}',
        '--seed=0',
        f'--save_dir={model_save_dir}',
        '--divergence_fn=approximate',
        '--log_level=INFO',
        '--use_wandb',
        '--lag=22',
        '${args}',
    ] if x is not None],
}

sweep_id = wandb.sweep(sweep_config)
print(dataset, predmodel, sweep_id)
wandb.agent(sweep_id)
