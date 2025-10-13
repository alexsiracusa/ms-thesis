import wandb
import torch
import numpy as np
import math
import random
from mnist.datasets import datasets, load_parquet
from mnist.util import train_mnist, create_model, sparse_perlin, flatten_images, sparse_random
from mnist.util.sizes import num_input, num_blocks
from torch.utils.data import DataLoader, TensorDataset


def train_model():
    run = wandb.init()
    dataset = run.config.dataset
    noise = run.config.noise
    epochs = run.config.epochs
    run_id = run.config.run_id

    # Load dataset
    artifact = run.use_artifact(f"{dataset}:latest")
    artifact_dir = artifact.download()

    train_images, train_labels = load_parquet(f"{artifact_dir}/train.parquet")
    test_images, test_labels = load_parquet(f"{artifact_dir}/train.parquet")

    train_images = flatten_images(train_images, kernel_size=(10,10), stride=10, padding=0)
    test_images = flatten_images(test_images, kernel_size=(10,10), stride=10, padding=0)

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    batch_size = math.ceil(len(train_dataset) / 200)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Train model
    if noise == 'sparse_random':
        p_random = 0.99 * random.random() + 0.01
        wandb.log({"p_random": p_random})
        densities = sparse_random((num_blocks - num_input, num_blocks), p_random=p_random)
    elif noise == 'sparse_perlin':
        clip = 0.99 * random.random() + 0.01
        wandb.log({"clip": clip})
        densities = sparse_perlin((num_blocks - num_input, num_blocks), clip=clip)
    output_size = datasets[dataset]
    model = create_model(densities)

    train_loss, test_loss = train_mnist(
        model, train_loader, test_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        epochs=epochs,
        output_size=output_size,
    )

    # Save results
    # np.save("density_map.npy", densities)
    # artifact_name = f"density_map_{wandb.run.id}"
    # artifact = wandb.Artifact(artifact_name, type="density_map")
    # artifact.add_file("density_map.npy")
    # wandb.log_artifact(artifact)

    densities = np.array(densities)
    wandb.run.summary["density_map"] = densities.tolist()

    wandb.log({"batch_size": batch_size})
    wandb.log({"average_density": densities.mean()})
    wandb.log({"train_loss": train_loss})
    wandb.log({"test_loss": test_loss})
