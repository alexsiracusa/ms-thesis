import wandb
from mnist.wandb_project import project_name
from mnist.datasets import datasets

# List of datasets
datasets = list(datasets.keys())
N = 2000
epochs = 5

# Sweep configuration
sweep_config = {
    'method': 'grid',  # iterate through all options
    'parameters': {
        'dataset': {
            'values': datasets
        },
        'noise': {
            'values': ['sparse_random', 'sparse_perlin']
        },
        'epochs': {
            'values': [epochs]
        },
        'run_id': {
            'values': list(range(N))
        }
    }
}

# Create the sweep
sweep_id = wandb.sweep(sweep_config, project=project_name)

