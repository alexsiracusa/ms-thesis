import wandb
from cifar10.compare_variance.wandb_project import project_name

N = 500
epochs = 5
batch_size = 128
block_sizes = [500, 300, 200, 100, 50]

# Sweep configuration
sweep_config = {
    'method': 'grid',  # iterate through all options
    'parameters': {
        'noise': {
            'values': ['sparse_random']
        },
        'block_size': {
            'values': block_sizes
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

