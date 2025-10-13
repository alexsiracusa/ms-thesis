import wandb
import numpy as np
import json
import matplotlib.pyplot as plt
from wandb_project import project_name

api = wandb.Api(timeout=60)
runs = api.runs(f"alexander-siracusa-worcester-polytechnic-institute/{project_name}", per_page=500)

test_losses = []
sum_densities = []

perlin_test_losses = []
perlin_sum_densities = []

print(len(runs))
for i, run in enumerate(runs[:2000]):
    print(i)
    try:
        summary = json.loads(run.summary._json_dict)
        config = json.loads(run.config)

        noise = config["noise"]["value"]
        densities = summary["density_map"]
        test_loss = summary["test_loss"]

        if noise == 'sparse_random':
            test_losses.append(test_loss)
            sum_densities.append(np.sum(densities))
        else:
            perlin_test_losses.append(test_loss)
            perlin_sum_densities.append(np.sum(densities))

        # if True:
        #     plt.imshow(densities, cmap="grey")
        #     plt.text(
        #         0, 0, f"{test_loss}",
        #         color='red', fontsize=8, weight='bold', ha='left', va='top'
        #     )
        #     plt.show()

    except Exception as e:
        print(f'error: {e}')
        continue

plt.scatter(
    sum_densities, test_losses,
    label='Sparse Random',
    alpha=0.5,
    s=5,
)
plt.scatter(
    perlin_sum_densities, perlin_test_losses,
    label='Sparse Perlin',
    alpha=0.5,
    s=5,
)

plt.xlabel('Num. Trainable Parameters')
plt.ylabel('Test Loss')
plt.legend(loc='upper right')
plt.show()

