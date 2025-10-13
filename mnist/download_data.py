import os

import wandb
import json
from wandb_project import project_name


def download_dataset(dataset, noise, data_dir='./data'):
    output_dir = f'{data_dir}/{dataset}'
    os.makedirs(output_dir, exist_ok=True)

    api = wandb.Api(timeout=60)
    runs = api.runs(
        path=f"alexander-siracusa-worcester-polytechnic-institute/{project_name}",
        filters={'config.dataset': dataset, 'config.noise': noise},
        per_page=50,
    )

    print(len(runs))

    arr = []

    for i, run in enumerate(runs):
        print(i)

        summary = json.loads(run.summary._json_dict)
        config = json.loads(run.config)

        data = {
            'test_loss': summary["test_loss"],
            'train_loss': summary["train_loss"],
            'epoch_losses': run.history(keys=['epoch_loss'])['epoch_loss'].to_numpy().tolist(),
            'epochs': config['epochs'],
            'dataset': config['dataset'],
            'batch_size': summary["batch_size"],
            'noise': config["noise"]["value"],
            'p_random': summary["p_random"] if "p_random" in summary else None,
            'clip': summary["clip"] if "clip" in summary else None,
            'average_density': summary["average_density"],
            'density_map': summary["density_map"],
        }

        arr.append(data)

    with open(f'{output_dir}/{noise}.txt', 'w') as f:
        for data in arr:
            f.write(json.dumps(data) + '\n')



if __name__ == '__main__':
    # download_dataset('mnist', 'sparse_random')
    # download_dataset('mnist', 'sparse_perlin')

    # download_dataset('emnist_letters', 'sparse_random')
    # download_dataset('emnist_letters', 'sparse_perlin')

    # download_dataset('emnist_balanced', 'sparse_random')
    # download_dataset('emnist_balanced', 'sparse_perlin')

    # download_dataset('kmnist', 'sparse_random')
    # download_dataset('kmnist', 'sparse_perlin')

    # download_dataset('fashion_mnist', 'sparse_random')
    # download_dataset('fashion_mnist', 'sparse_perlin')

    # download_dataset('cifar10', 'sparse_random')
    # download_dataset('cifar10', 'sparse_perlin')

    # --------

    download_dataset('sign_mnist', 'sparse_random')
    download_dataset('sign_mnist', 'sparse_perlin')

    download_dataset('chinese_mnist', 'sparse_random')
    download_dataset('chinese_mnist', 'sparse_perlin')

    download_dataset('kannada_mnist', 'sparse_random')
    download_dataset('kannada_mnist', 'sparse_perlin')

    download_dataset('dig_mnist', 'sparse_random')
    download_dataset('dig_mnist', 'sparse_perlin')

    download_dataset('overhead_mnist', 'sparse_random')
    download_dataset('overhead_mnist', 'sparse_perlin')

