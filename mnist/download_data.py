import wandb
import numpy as np
import os

api = wandb.Api()
runs = api.runs("alexander-siracusa-worcester-polytechnic-institute/density-map-test")

for run in runs:
    loss = run.summary.get("test_loss")
    for artifact in run.logged_artifacts():
        if artifact.name.startswith("density_map"):
            artifact_dir = artifact.download()
            arr = np.load(os.path.join(artifact_dir, "density_map.npy"))
            print(run.name, loss, arr.shape)
