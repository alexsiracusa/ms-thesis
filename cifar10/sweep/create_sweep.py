import wandb
from cifar10.sweep.wandb_project import project_name

N = 2000
epochs = 5
batch_size = 128

# Sweep configuration
sweep_config = {
    'method': 'grid',  # iterate through all options
    'parameters': {
        'noise': {
            'values': ['sparse_random', 'sparse_perlin', 'sparse_perlin_clip']
        },
        'epochs': {
            'values': [epochs]
        },
        'batch_size': {
            'values': [batch_size]
        },
        'run_id': {
            'values': list(range(N))
        }
    }
}

# Create the sweep
sweep_id = wandb.sweep(sweep_config, project=project_name)

