import os
import wandb
import json
from mnist.dataset_features.wandb_project import project_name


def download_dataset(data_dir='../data'):
    output_dir = f'{data_dir}'
    os.makedirs(output_dir, exist_ok=True)

    api = wandb.Api(timeout=60)
    runs = api.runs(
        path=f"alexander-siracusa-worcester-polytechnic-institute/{project_name}",
        filters={'sweep': 'x3i2vbjf'},
        per_page=50,
    )

    print(len(runs))

    for i, run in enumerate(runs):
        print(i)

        if run.state != 'finished':
            continue

        summary = json.loads(run.summary._json_dict)
        config = json.loads(run.config)

        features = {
            'num_classes': summary["num_classes"],
            'loss_uniform': summary["loss_uniform"],
            'lr_loss': summary["lr_loss"],
            'nn_loss_0.1_test': summary["nn_loss_0.1_test"],
            'nn_loss_0.1_train': summary["nn_loss_0.1_train"],
            'nn_loss_0.25_test': summary["nn_loss_0.25_test"],
            'nn_loss_0.25_train': summary["nn_loss_0.25_train"],
            'nn_loss_0.5_test': summary["nn_loss_0.5_test"],
            'nn_loss_0.5_train': summary["nn_loss_0.5_train"],
        }

        with open(f'{data_dir}/{config["dataset"]["value"]}/dataset_features.json', 'w') as f:
            json.dump(features, f, indent=4)



if __name__ == '__main__':
    download_dataset()

