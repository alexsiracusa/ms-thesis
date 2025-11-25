import wandb
from mnist.wandb_project import project_name
from mnist.datasets import datasets

# List of datasets
datasets = list(datasets.keys())

# Sweep configuration
sweep_config = {
    'method': 'grid',  # iterate through all options
    'parameters': {
        'dataset': {
            'values': datasets
        }
    }
}

# Create the sweep
sweep_id = wandb.sweep(sweep_config, project=project_name)

