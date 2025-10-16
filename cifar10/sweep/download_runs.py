import os

import wandb
import json
from cifar10.sweep.wandb_project import project_name


def download_dataset(noise, data_dir='../data'):
    output_dir = f'{data_dir}'
    os.makedirs(output_dir, exist_ok=True)

    api = wandb.Api(timeout=60)
    runs = api.runs(
        path=f"alexander-siracusa-worcester-polytechnic-institute/{project_name}",
        filters={'config.noise': noise},
        per_page=50,
    )

    print(len(runs))

    arr = []

    for i, run in enumerate(runs):
        print(i)

        if run.state != 'finished':
            continue

        summary = json.loads(run.summary._json_dict)
        config = json.loads(run.config)

        data = {
            'epochs': config['epochs'],
            'batch_size': config["batch_size"],
            'noise': config["noise"]["value"],
            'test_losses': summary["test_losses"],
            'train_losses': summary["train_losses"],
            'epoch_losses': summary["epoch_losses"],
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
    download_dataset('sparse_random')
    # download_dataset('sparse_perlin')
    # download_dataset('sparse_perlin_clip')

