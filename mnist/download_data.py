import wandb
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from wandb_project import project_name

api = wandb.Api()
runs = api.runs(f"alexander-siracusa-worcester-polytechnic-institute/{project_name}")

test_losses = []
sum_densities = []

perlin_test_losses = []
perlin_sum_densities = []

for run in runs[:200]:
    try:
        noise_value = json.loads(run.config)['noise']['value']
        final_test_loss = run.history(keys=["test_loss"])["test_loss"].iloc[-1]

        for artifact in run.logged_artifacts():
            if artifact.name.startswith("density_map"):
                artifact_dir = artifact.download(skip_cache=False)
                arr = np.load(os.path.join(artifact_dir, 'density_map.npy'))

                if noise_value == 'sparse_random':
                    test_losses.append(final_test_loss)
                    sum_densities.append(np.sum(arr))
                else:
                    perlin_test_losses.append(final_test_loss)
                    perlin_sum_densities.append(np.sum(arr))

                # print(final_test_loss, np.sum(arr))

                # plt.imshow(arr, cmap="grey")
                # plt.text(
                #     0, 0, f"{final_test_loss}",
                #     color='red', fontsize=8, weight='bold', ha='left', va='top'
                # )
                # plt.show()
    except:
        continue

plt.scatter(
    sum_densities, test_losses,
    label='Sparse Random',
    alpha=0.5,
    s=5,
)
plt.scatter(
    perlin_test_losses, perlin_test_losses,
    label='Sparse Perlin',
    alpha=0.5,
    s=5,
)

plt.xlabel('Num. Trainable Parameters')
plt.ylabel('Test Loss')
plt.show()

