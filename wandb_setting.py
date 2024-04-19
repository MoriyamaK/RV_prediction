import os
import getpass
import wandb

wandb_dir = f'/tmp/wandb_{getpass.getuser()}'
os.makedirs(wandb_dir, exist_ok=True)

settings = {
    'WANDB_ENTITY': 'acc_name',    # replace this with your WANDB account name
    'WANDB_DIR': wandb_dir,
    'WANDB_PROJECT': 'project_name',  # you can change this to the name you like 
    'WANDB_API_KEY': 'wandb_api_key',# replace this with your WANDB API KEY
}

def config():
    for k, v in settings.items():
        os.environ[k] = v
