import wandb
import numpy as np
import os
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("alexander-siracusa-worcester-polytechnic-institute/density-map-test")

for run in runs:
    history = run.history(keys=["test_loss"])
    if not history.empty:
        final_test_loss = history["test_loss"].iloc[-1]  # last logged value
    else:
        final_test_loss = None

    for artifact in run.logged_artifacts():
        if artifact.name.startswith("density_map"):
            artifact_dir = artifact.download()
            arr = np.load(os.path.join(artifact_dir, "density_map.npy"))

            print(final_test_loss, np.sum(arr))

            # plt.imshow(arr, cmap="grey")
            # plt.text(
            #     0, 0, f"{final_test_loss}",
            #     color='red', fontsize=8, weight='bold', ha='left', va='top'
            # )
            # plt.show()
