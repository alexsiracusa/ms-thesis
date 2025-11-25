import wandb
import torch
import math
from mnist.datasets import datasets, load_parquet
from mnist.util import train_mnist, create_model, sparse_perlin, flatten_images, sparse_random
from mnist.util.sizes import num_input, num_blocks
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def wandb_run():
    run = wandb.init()
    dataset = run.config.dataset

    # Load dataset
    artifact = run.use_artifact(f"{dataset}:latest")
    artifact_dir = artifact.download()

    train_images, train_labels = load_parquet(f"{artifact_dir}/train.parquet")
    test_images, test_labels = load_parquet(f"{artifact_dir}/train.parquet")

    train_images = flatten_images(train_images, kernel_size=(10, 10), stride=10, padding=0)
    test_images = flatten_images(test_images, kernel_size=(10, 10), stride=10, padding=0)

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    batch_size = math.ceil(len(train_dataset) / 200)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    #  Static Features
    num_classes = datasets[dataset]
    wandb.log({"num_classes": num_classes})
    wandb.log({"loss_uniform": -np.log(1 / num_classes)})

    # Logistic Regression
    model = LogisticRegression(max_iter=500)
    model.fit(train_images, train_labels)

    probs = model.predict_proba(test_images)
    ce_loss = log_loss(test_labels, probs)

    wandb.log({"lr_loss": ce_loss})

    # Model features
    for density in [0.1, 0.25, 0.5]:
        model = create_model(density)

        train_loss, test_loss = train_mnist(
            model, train_loader, test_loader,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            epochs=5,
            output_size=num_classes,
        )

        wandb.log({f"nn_loss_{density}_test": test_loss})
        wandb.log({f"nn_loss_{density}_train": train_loss})
