import wandb
import numpy as np


wandb.init(
    project="density-map-test",
    config={
        "dataset": "CIFAR10",
    }
)

density_map = np.random.rand(100, 100)
test_loss = np.random.rand()

np.save("density_map.npy", density_map)

artifact_name = f"density_map_{wandb.run.id}"
artifact = wandb.Artifact(artifact_name, type="density_map")
artifact.add_file("density_map.npy")
wandb.log_artifact(artifact)

wandb.log({"test_loss": test_loss})
wandb.finish()